#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        //int normType = cv::NORM_HAMMING;
        int normType  = descriptorType.compare("DES_BINARY")==0 ? cv::NORM_HAMMING : cv::NORM_L2;
        descSource.convertTo(descSource, CV_8U);
        descRef.convertTo(descRef, CV_8U);
        std::cout<<" I was here" <<std::endl;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {

         // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);

        matcher =cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher->knnMatch(descSource,descRef,knn_matches,2);

        const float ratio_threshold = 0.8f;
        for (size_t i=0; i<knn_matches.size();i++)
        {
            if (knn_matches[i][0].distance<ratio_threshold * knn_matches[i][1].distance)
            {
                matches.push_back(knn_matches[i][0]);
            }

        }
    }


}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // TODO BRIEF, ORB, FREAK, AKAZE, SIFT Types
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF")==0)
    {
        extractor = cv::BRISK::create();

    }
    else if (descriptorType.compare("ORB")==0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK")==0)
    {
        cv::xfeatures2d::FREAK extractor;
        // extractor = cv::xfeatures2d::FREAK::create(); // <-- Alternative version for Udacity VM

    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();


    }
    else if (descriptorType.compare("SIFT")==0)
    {

        extractor = cv::SIFT::create(); // same thing as below
        //extractor = cv::xfeatures2d::SiftDescriptorExtractor::create(); // <-- This one is correct
        cv::Mat img8U,imgGray;
        std::cout<<"Old Img type:" <<img.type()<<std::endl;
        img.convertTo(img8U, CV_8U);
        //cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        std::cout<<"New Img type:" <<img8U.type()<<std::endl;

    }
    else if (descriptorType.compare("SURF")==0)
    {
        //extractor = cv::xfeatures2d::SURF::create();
        int minHessian = 400;
        //extractor = cv::xfeatures2d::SurfDescriptorExtractor::create(minHessian); // <-- This one is correct
    }
    else
    {
        std::cout<<"Wrong DescriptorType name :  "<<descriptorType<<std::endl;
    }



    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

    //Detector Parameters;
    int blockSize = 2;
    int apertureSize = 3 ;
    int minResponse = 100;
    double k = 0.04;

    //Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);  // Scales, calculates absolute values, and converts the result to 8-bit.

    // The brighter a pixel, the higher the Harris corner response.
    // and perform a non-maximum suppression (NMS) in a local neighborhood around
    // each maximum. The resulting coordinates shall be stored in a list of keypoints
    // of the type `vector<cv::KeyPoint>`.

    cv::Mat nms_image = dst_norm.clone();
    int threshold =100;
    cv::Mat mask_for_nms;

    //std::vector<cv::KeyPoint> kpts;
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression

    for (int r =0; r<nms_image.rows; r++)
    {
        for (int c = 0; c<nms_image.cols; c++)
        {
            if ((float) nms_image.at<float>(r,c) > threshold)
            {
                //cv::circle(nms_image,cv::Point(r,c),10,cv::Scalar(0),10,8,0);

                cv::KeyPoint newKeyPoint;
                newKeyPoint.size = 2*apertureSize;  // diameter of the meaningful keypoint neighborhood
                newKeyPoint.pt = cv::Point2f(c,r);  // coordinates of the keypoints
                newKeyPoint.response =  (float) nms_image.at<float>(r,c);

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it !=keypoints.end(); it++)
                {
                    double kptsOverlap = cv::KeyPoint::overlap(newKeyPoint,*it);
                    if(kptsOverlap>maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response> (*it).response)
                        {                       // if overlap is >t AND response is higher for new kpt
                            *it =newKeyPoint;   // replace old key point with new one
                            break;       // quit loop over keypoints

                        }
                    }
                }
                if (!bOverlap)
                {                                    // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint);     // store new keypoint in dynamic list

                }

            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if(bVis)
    {
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName,cv::WINDOW_AUTOSIZE);
        cv::imshow(windowName, nms_image);
        cv::drawKeypoints(dst_norm_scaled,keypoints,nms_image,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName,nms_image);
        cv::waitKey(0);
    }

}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    //FAST, BRISK, ORB, AKAZE, SIFT

    if(detectorType.compare("FAST") == 0)
    {
        double t = (double)cv::getTickCount();
        cv::FAST(img,keypoints,true);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "Fast with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if(detectorType.compare("BRISK")==0)
    {

        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "BRISK detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if (detectorType.compare("ORB")==0)
    {
        double t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "ORB detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;


    }
    else if (detectorType.compare("AKAZE")==0)
    {
        double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
        double t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector =cv::AKAZE::create();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "AKAZE detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;


    }
    else if (detectorType.compare("SIFT") ==0)
    {
        //cv::Ptr<cv::FeatureDetector> sift_detector = cv::xfeatures2d::SIFT::create(); // <-- Bu dogru
        cv::Ptr<cv::FeatureDetector> sift_detector = cv::SIFT::create();

        double t = (double)cv::getTickCount();
        cv::Mat img8U,imgGray;
        //img.convertTo(img8U, CV_8U);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
        sift_detector->detect(imgGray,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "SIFT detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else
    {
        std::cout<<"Wrong detectorType name"<<endl;
    }

    if(bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img,keypoints,visImage,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(detectorType,1);
        cv::imshow(detectorType,visImage);
    }

}