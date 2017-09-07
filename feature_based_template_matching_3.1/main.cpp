#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>		// FlannBasedMatcher
#include <opencv2/xfeatures2d.hpp>		// SIFT

#define TEMPLATE_PATH			"[some path]"
#define SOURCE_PATH				"[some path]"

int main()
{
	//-- Step 0: Load template and source images with grayscale
	cv::Mat tpl = cv::imread(TEMPLATE_PATH, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat src = cv::imread(SOURCE_PATH, CV_LOAD_IMAGE_GRAYSCALE);

	//-- Step 1: Detect the keypoints using SIFT detector
	cv::Ptr<cv::Feature2D> sift_detector = cv::xfeatures2d::SiftFeatureDetector::create();
	std::vector<cv::KeyPoint> tpl_kpts, src_kpts;
	sift_detector->detect(tpl, tpl_kpts);
	sift_detector->detect(src, src_kpts);

	//-- Step 2: Calculate descriptors
	cv::Ptr<cv::DescriptorExtractor> sift_extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
	cv::Mat tpl_desc, src_desc;
	sift_extractor->compute(tpl, tpl_kpts, tpl_desc);
	sift_extractor->compute(src, src_kpts, src_desc);

	//-- Step 3: Match descriptor vectors using FLANN matcher
	cv::Ptr<cv::DescriptorMatcher> flann_matcher = cv::FlannBasedMatcher::create("FlannBased");
	std::vector<cv::DMatch> flann_matches;
	flann_matcher->match(tpl_desc, src_desc, flann_matches);

	//-- Step 4: Find "good" matches
	double min_dist = 100;
	for (int i = 0; i < tpl_desc.rows; i++) {
		double dist = flann_matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
	}

	printf("-- Min dist : %f\n", min_dist);

	std::vector<cv::DMatch> good_matches;
	for (cv::DMatch match : flann_matches)
		if (match.distance <= cv::max(5 * min_dist, 0.02))
			good_matches.push_back(match);

	//-- Step 5: Draw "good" matches
	cv::Mat out;
	cv::drawMatches(tpl, tpl_kpts, src, src_kpts, good_matches, out,
		cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	cv::imshow("out", out);

	for (cv::DMatch match : good_matches)
		printf("-- Keypoint 1: %d  -- Keypoint 2: %d -- dist: %f\n", match.queryIdx, match.trainIdx, match.distance);
	cv::waitKey(0);

	return 0;
}