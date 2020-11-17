// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(const std::vector<cv::KeyPoint>& kptsPrev, const std::vector<cv::KeyPoint>& kptsCurr,
                      const std::vector<cv::DMatch>& kptMatches, double frameRate, double& TTC) {
// compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; it1++) {

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); it2++) {

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

// only continue if list of distance ratios is not empty
    if (distRatios.empty()) {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());

    std::cout << "Distance Ratios:" << std::endl;
    for (const auto& dist : distRatios) {
        std::cout << dist << " ";
    }
    std::cout << std::endl;


    long medIndex = floor(distRatios.size() / 2.0);
// compute median dist. ratio to remove outlier influence
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];

    std::cout << "medDistRatio = " << medDistRatio << std::endl;

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);

}

void clusterHelper(int index, const std::vector<std::vector<float>>& points, std::vector<int>& cluster, std::vector<bool>& processed,
                   const std::shared_ptr<KdTree>& tree, float distanceTol) {
    processed[index] = true;
    cluster.push_back(index);

    std::vector<int> nearest = tree->search(points[index], distanceTol);

    for (int id : nearest) {
        if (!processed[id]) {
            clusterHelper(id, points, cluster, processed, tree, distanceTol);
        }
    }

}

std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<float>>& points,
                                               const std::shared_ptr<KdTree>& tree, float distanceTol) {

    std::vector<std::vector<int>> clusters;
    std::vector<bool> processed(points.size(), false);

    int i = 0;
    while (i < points.size()) {
        if (processed[i]) {
            i++;
            continue;
        }

        std::vector<int> cluster;
        clusterHelper(i, points, cluster, processed, tree, distanceTol);
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

void computeTTCLidar(const std::vector<LidarPoint> &lidarPointsPrev,
                     const std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
// auxiliary variables
    double dT = 1 / frameRate; // time between two measurements in seconds
    constexpr double laneWidth = 4.0; // assumed width of the ego lane
    constexpr float clusterTolerance = 0.1;

// find closest distance to LiDAR points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;

    std::cout << "Process previous frame..." << std::endl;
    std::vector<LidarPoint> lidarPointsPrevClustered = removeLidarOutlier(lidarPointsPrev, clusterTolerance);

    std::cout << "Process current frame..." << std::endl;
    std::vector<LidarPoint> lidarPointsCurrClustered = removeLidarOutlier(lidarPointsCurr, clusterTolerance);


    for (const auto & it : lidarPointsPrevClustered) {
        if (abs(it.y) <= laneWidth / 2.0) { // 3D point within ego lane?
            minXPrev = it.x < minXPrev ? it.x : minXPrev;
        }
    }

    for (const auto & it : lidarPointsCurrClustered) {
        if (abs(it.y) <= laneWidth / 2.0) { // 3D point within ego lane?
            minXCurr = it.x < minXCurr ? it.x : minXCurr;
        }
    }

    std::cout << "Prev min X = " << minXPrev << std::endl;
    std::cout << "Curr min X = " << minXCurr << std::endl;

// compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);

}




template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType>& x) {
    using pairtype = std::pair<KeyType, ValueType>;
    return *std::max_element(x.begin(), x.end(), [] (const pairtype & p1, const pairtype & p2) {
        return p1.second < p2.second;
    });
}
