
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

// Structure to represent node of kd tree
struct Node
{
    std::vector<float> point;
    int id;
    Node* left;
    Node* right;

    Node(std::vector<float> arr, int setId)
            :	point(arr), id(setId), left(NULL), right(NULL)
    {}
};

struct KdTree
{
    Node* root;

    KdTree()
            : root(NULL)
    {}

    void insertHelper(Node** node, uint depth, std::vector<float> point, int id) // passing memory address of pointer
    {
        if (*node == NULL)
        {
            *node = new Node (point,id); // assign a brand new node.
        }
        else
        {
            // Calculate current dim
            uint cd = depth %2; // two is because we are working with 2D case. The depth is even or odd
            // cd will be either 0 or 1
            if (point[cd]< ((*node)->point[cd]))// if it is even.
            {
                insertHelper(&((*node)->left),depth+1, point,id);
            }
            else
                insertHelper(&((*node)->right),depth+1,point,id);
        }


    }

    void insert(std::vector<float> point, int id)
    {
        // TODO: Fill in this function to insert a new point into the tree
        // the function should create a new node and place correctly with in the root
        insertHelper(&root,0,point,id); // passing memory address for a root.

    }
    void searchHelper (std::vector<float> target, Node* node, int depth, float distanceTol, std::vector<int>& ids)
    {
        if (node!= NULL)
        {
            if( (node->point[0]>=(target[0]-distanceTol) && node->point[0]<=(target[0]+distanceTol)) &&
                (node->point[1]>=(target[1]-distanceTol) && node->point[1]<=(target[1]+distanceTol)) )
            {
                float distance = sqrt((node->point[0] - target[0]) * (node->point[0]-target[0])+ (node->point[1] - target[1]) * (node->point[1] - target[1]));
                if (distance <= distanceTol)
                    ids.push_back(node->id);
            }

            // check across boundary
            if ((target[depth%2] - distanceTol)<node->point[depth%2])
                searchHelper(target,node->left,depth+1,distanceTol,ids);
            if ((target[depth%2] + distanceTol)>node->point[depth%2])
                searchHelper(target,node->right,depth+1, distanceTol,ids);
        }
    }

    // return a list of point ids in the tree that are within distance of target
    std::vector<int> search(std::vector<float> target, float distanceTol) // we passing target which is represented as X and Y in this case, and distanceTolerance with that target.
    {
        // TODO: Looking for ids which close to target (X,Y) and around distanceTolerance
        std::vector<int> ids;
        searchHelper(target,root,0,distanceTol, ids);
        return ids;
    }


};


#endif /* dataStructures_h */
