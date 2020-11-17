
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
//#include "kdtree.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow( windowName, cv::WINDOW_FREERATIO);
    cv::resizeWindow(windowName,640,640);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Previous = Query
    // Current = Train

    /*Calculation of Euclidean distance between between two matched kpts frames which are in RIO */

    std::vector<float> EuclideanDistance;
    float maxDistanceFromMedian = 2.0;

    for (const auto& kptMatche : kptMatches)
    {
        const auto& idxCurr = kptMatche.trainIdx;
        const auto& idxPrev = kptMatche.queryIdx;

        if (boundingBox.roi.contains(kptsCurr[idxCurr].pt))
        {
            double dist = cv::norm(kptsPrev[idxPrev].pt - kptsCurr[idxCurr].pt);
            EuclideanDistance.push_back(dist);
        }
    }

    // Euclidean Distance Mean
    float EuclDistMean   = (float) accumulate(EuclideanDistance.begin(),EuclideanDistance.end(),0.0)/ EuclideanDistance.size();

    // Euclidean Distance Meadian
    std::sort(EuclideanDistance.begin(), EuclideanDistance.end());
    int size = EuclideanDistance.size();
    float EuclDistMedian = EuclideanDistance.size() %2 == 0 ? (EuclideanDistance[size/ 2.0] + EuclideanDistance[size/ 2 -1]) /2 : EuclideanDistance[size/2];

    for (const auto& kptMatche : kptMatches)
    {
            if (boundingBox.roi.contains(kptsCurr[kptMatche.trainIdx].pt))
            {
                double temp_dist = cv::norm(kptsCurr[kptMatche.trainIdx].pt - kptsPrev[kptMatche.queryIdx].pt);
                // if bigger than Euclidean distance and Median values then add to BB.
                //if ((temp_dist > (EuclDistMedian - maxDistanceFromMedian)) && (temp_dist < (EuclDistMedian + maxDistanceFromMedian)) && (temp_dist < EuclDistMean*1.5))
                if (temp_dist < EuclDistMean*1.5)
                {
                    boundingBox.keypoints.push_back(kptsCurr[kptMatche.trainIdx]);
                    boundingBox.kptMatches.push_back(kptMatche);
                }
            }
    }
    std::cout << " Mean value of Euclidean Distance: " << EuclDistMean << "\n Before filtering there are: " << size << "\n After filtering, there are " << boundingBox.keypoints.size() << std::endl;

}

/**/
// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(const std::vector<cv::KeyPoint>& kptsPrev, const std::vector<cv::KeyPoint>& kptsCurr,
                      const std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // Compute distance ration between all matched keypoints
    std::vector<double> distRatios; // stores the distance rations for all kpts between curr, and prev. fram
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop
            double minDist = 100.0;    // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.empty())
    {
        TTC = NAN;
        return;
    }

    /* Calculation Median of distance Ratio */
    std::sort(distRatios.begin(), distRatios.end());
    std::cout << "Distance Ratios:" << std::endl;

    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio  = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    /* Calculation of Mean distance Ratio */
    // int size = distRatios.size();
    // double meanDistRatio = size % 2 == 0 ? meanDistRatio =  (distRatios[size/ 2] + distRatios[size / 2 -1])  : distRatios[size/2]; // will lead a faulty calculation of the TTC

    std::cout << "medDistRatio = " << medDistRatio << std::endl;

    /*TTC*/
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);

}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{

    // auxiliary variables
    double dT = 1.0/frameRate;        // time between two measurements in seconds
    constexpr double  laneWidth = 4.0; // assumed width of the ego lane
    constexpr float clusterTolerance = 0.1;

    // According to Udacity Suggestion
    std::cout << "Process previous frame..." << std::endl;
    std::vector<LidarPoint> lidarPointsPrevClustered = removeLidarOutlier(lidarPointsPrev, clusterTolerance);

    std::cout << "Process current frame..." << std::endl;
    std::vector<LidarPoint> lidarPointsCurrClustered = removeLidarOutlier(lidarPointsCurr, clusterTolerance);

    // PreProcessing of LidarPoints in order to get rid of random points
    std::cout<<"Apply KD-Tree and Euclidean cluster based on distance threshold and min cluster size"<<std::endl;
    std::cout<<"LidarPointsPrev"<<std::endl;
    preProcessing(lidarPointsPrev,clusterTolerance);
    std::cout<<"lidarPointsCurr"<<std::endl;
    preProcessing(lidarPointsCurr,clusterTolerance);

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (const auto & it : lidarPointsPrevClustered)
    {
        if(abs(it.y)<=laneWidth/2)
        {   // 3D point within ego lane?
            // this is a conditional operation (condition) ? expression1 : expression2
            // if condition ( in our case minXPrev> it-x) true than expression1 (in our case it->x),
            // if false then expression2 ( in our case minXPrev)
            minXPrev = it.x < minXPrev ? it.x : minXPrev;
        }
    }

    //for (auto it = lidarPointsCurrClustered.begin(); it != lidarPointsCurrClustered.end(); ++it)
    for (const auto & it : lidarPointsCurrClustered)
    {
        if(abs(it.y)<=laneWidth/2)
        {   // 3D point within ego lane?
            // this is a conditional operation (condition) ? expression1 : expression2
            // if condition ( in our case minXCurr> it-x) true than expression1 (in our case it->x),
            // if false then expression2 ( in our case minXCurr)
            minXCurr = minXCurr > it.x ? it.x : minXCurr;
        }
    }
    std::cout << "Prev min X = " << minXPrev << std::endl;
    std::cout << "Curr min X = " << minXCurr << std::endl;

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);  // TTC = (d1/v0) = (d1*frameRate) / (d0-d1)
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    // PS: we already removed Lidar points based on distance properties,
    //     and so that we focus only  on ego lane (front vehicle)

    // You need to verify when you try to match previous bounding boxes to current bounding boxes,
    // you check if the previous bounding box contains prevKeyPoint and current bounding box contains currKeyPoint at the same time.
    // If so, the count number of total matches between previous bounding box ID and current bounding box ID will be accumulated(+1)

    int prevBoxID;
    int currBoxID;
    cv::Mat count = cv::Mat::zeros(prevFrame.boundingBoxes.size(), currFrame.boundingBoxes.size(), CV_32S);

    // loop over all matched descriptors
    for (auto descr = matches.begin(); descr != matches.end(); ++descr)
    {
        // Loop over all previous bounding boxes
        for (auto pBox = prevFrame.boundingBoxes.begin(); pBox != prevFrame.boundingBoxes.end(); ++pBox)
        {
            // Get the matching key-points index for each of the images
            int idxPrevious = descr->queryIdx;

            // find if matched descriptors (keypoints) is in previous bounding box
            if (pBox->roi.contains(prevFrame.keypoints[idxPrevious].pt) )
            {
                // store ID of previous bounding box, for late matching
                prevBoxID = pBox->boxID;

                // Loop over current bounding boxes
                for (auto cBox = currFrame.boundingBoxes.begin(); cBox != currFrame.boundingBoxes.end(); ++cBox)
                {
                    int idxCurrent  = descr->trainIdx;
                    // find if matched descriptors (keypoints) is in current bounding box
                    if (cBox->roi.contains(currFrame.keypoints[idxCurrent].pt))
                    {
                        currBoxID = cBox->boxID;
                        count.at<int>(prevBoxID,currBoxID) = count.at<int>(prevBoxID,currBoxID) + 1;
                    }
                }

            }
        }

    }
    //std::cout << "count = " <<  std::endl << " "  << count <<  std::endl <<  std::endl;

    // for each prev bb find and compare the max count of corresponding curr bb.
    // the curr bb with max no. of matches (max count) is the bbestmatch

    // Associate the bounding boxes with the highest number of occurrences
    // ps: row  -> previous BB
    //   : cols -> current  BB
    for (size_t r = 0; r < count.rows; r++ )
    {
        int id = -1;
        int maxvalue = 0;
        for (size_t c = 0; c < count.cols; c++)
        {
            if (count.at<int>(r,c) > maxvalue && count.at<int>(r,c) > 0  )
            {
                maxvalue  = count.at<int>(r,c);
                id = c;
            }
        }
        if (id!= -1)
        {
            std::cout <<r<< " " << id <<std::endl;
            bbBestMatches[r] = id;    // or bbBestMatches.insert({r, id});

        }
    }
}


//void clusterHelper(int index, const std::vector<std::vector<float>>& points, std::vector<int>& cluster, std::vector<bool>& processed, KdTree* tree, float distanceTol )
void clusterHelper(int index,   const std::vector<std::vector<float>>& points, std::vector<int>& cluster, std::vector<bool>& processed, const std::shared_ptr<KdTree>& tree, float distanceTol)

{
    processed[index] =true;
    cluster.push_back(index);

    std::vector<int> nearest = tree->search(points[index],distanceTol);

    for (int id :nearest)
    {
        if (!processed[id])
            clusterHelper(id,points,cluster,processed,tree,distanceTol);

    }
}

std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<float>>& points,
                                               const std::shared_ptr<KdTree>& tree, float distanceTol)
{
    // TODO: Fill out this function to return list of indices for each cluster

    std::vector<std::vector<int>> clusters;
    std::vector<bool> processed(points.size(),false);

    int i = 0;
    while (i<points.size())
    {
        if (processed[i])
        {
            i++;
            continue;
        }
        std::vector<int>cluster;
        clusterHelper(i,points,cluster,processed,tree,distanceTol);
        clusters.push_back(cluster);
        i++;
    }

// Return list of indices for each cluster
    return clusters;
}

std::vector<LidarPoint> removeLidarOutlier(const std::vector<LidarPoint> &lidarPoints, float clusterTolerance) {
    auto treePrev = std::make_shared<KdTree>();
    std::vector<std::vector<float>> points;
    for (int i=0; i< lidarPoints.size(); i++) {
        std::vector<float> point({static_cast<float>(lidarPoints[i].x),
                                  static_cast<float>(lidarPoints[i].y),
                                  static_cast<float>(lidarPoints[i].z)});
        points.push_back(point);
        treePrev->insert(points[i], i);
    }
    std::vector<std::vector<int>> cluster_indices = euclideanCluster(points, treePrev, clusterTolerance);

    std::vector<LidarPoint> maxLidarPointsCluster;
    for (const auto& get_indices : cluster_indices) {
        std::vector<LidarPoint> temp;
        for (const auto index : get_indices) {
            temp.push_back(lidarPoints[index]);
        }
        std::cout << "Cluster size = " << temp.size() << std::endl;
        if (temp.size() > maxLidarPointsCluster.size()) {
            maxLidarPointsCluster = std::move(temp);
        }
    }
    std::cout << "Max cluster size = " << maxLidarPointsCluster.size() << std::endl;
    return maxLidarPointsCluster;
}


template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType>& x) {
    using pairtype = std::pair<KeyType, ValueType>;
    return *std::max_element(x.begin(), x.end(), [] (const pairtype & p1, const pairtype & p2) {
        return p1.second < p2.second;
    });
}

// Support function for kd_tree and Euclidean clustering
void preProcessing (std::vector<LidarPoint> &lidarPoints,float distanceTol,int minSize)
{
    std::cout << "PointCloud representing the Cluster before PreProcessing: " << lidarPoints.size () << " data points." << std::endl;
    // Pre-processing: Implementation of KD_Tree and Euclidean Clustering Algorithm based on Lidar Course

    // Initial clustering parameters;
    int maxSize = 1000;

    //KdTree* tree = new KdTree;
    auto tree = std::make_shared<KdTree>();

    std::vector<std::vector<LidarPoint>> clusters;  // My version
    //std::vector<LidarPoint> clusters;

    std::vector<std::vector<float>> points;

    for (int i = 0 ; i<lidarPoints.size(); ++i)
    {
        // convert double to float
        float x = (float) lidarPoints[i].x;
        float y = (float) lidarPoints[i].y;
        float z = (float) lidarPoints[i].z;

        std::vector<float> point({x,y,z});
        points.push_back(point);
        tree->insert(points[i],i);
    }
    std::vector<std::vector<int>> clusters_Indicies = euclideanCluster(points, tree, distanceTol);


    for (const auto& getIndices: clusters_Indicies)
    {
        std::vector<LidarPoint> cloudCluster;

        for (const auto index: getIndices)
        {
            cloudCluster.push_back(lidarPoints[index]);
        }

        if (cloudCluster.size() >= minSize && cloudCluster.size()<=maxSize)
        {
            // My version
            lidarPoints = cloudCluster;
            clusters.push_back(cloudCluster);
            //clusters = std::move(cloudCluster);

        }

    }
    std::cout << "PointCloud representing the Cluster after  PreProcessing: " << lidarPoints.size () << " data points." << std::endl;

}

