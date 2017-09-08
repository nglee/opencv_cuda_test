#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>		// FlannBasedMatcher, KAZE, AKAZE
#include <opencv2/xfeatures2d.hpp>		// SIFT, SURF

#include "main.h"

#define TEMPLATE_PATH           // write template path here
#define SOURCE_PATH             // write source path here

cv::Mat tpl;
cv::Mat src;

// Initial Settings
enum DetectorType selectedDetector = DETECTOR_SIFT, prevDetector;
enum MatcherType selectedMatcher = MATCHER_BF, prevMatcher;
enum CommandType selectedCommand = CMD_DETECT_AND_MATCH, prevCommand;

// Stringified names for enumerations for convenience
const std::string detector_name[DETECTOR_SIZE] = { "SIFT", "SURF", "KAZE", "AKAZE", "ORB" };
const std::string matcher_name[MATCHER_SIZE] = { "FLANN", "Brute Force" };
const std::string command_name[CMD_SIZE] = { "Detect", "Detect and match" };

void draw()
{
	prevDetector = selectedDetector;
	prevMatcher = selectedMatcher;
	prevCommand = selectedCommand;

	printf("\n\n------------------------------ REDRAW ------------------------------\n");
	printf("-- Selected feature detecting method : %s\n", detector_name[selectedDetector].c_str());
	
	if (selectedCommand != CMD_DETECT) {
		printf("-- Selected matching method : %s\n", matcher_name[selectedMatcher].c_str());
		printf("-- Selected command : %s\n", command_name[selectedCommand].c_str());
	}

	cv::Ptr<cv::Feature2D> detector;
	switch (selectedDetector) {
	case DETECTOR_SIFT: detector = cv::xfeatures2d::SiftFeatureDetector::create(); break;
	case DETECTOR_SURF: detector = cv::xfeatures2d::SurfFeatureDetector::create(); break;
	case DETECTOR_KAZE: detector = cv::KAZE::create(); break;
	case DETECTOR_AKAZE: detector = cv::AKAZE::create(); break;
	case DETECTOR_ORB: detector = cv::ORB::create(); break;
	default: printf("Unsupported choice of feature detector(%d), exiting...\n", selectedDetector); exit(1);
	}

	//-- Step 1: Detect keypoints and compute descriptors
	std::vector<cv::KeyPoint> tpl_kpts, src_kpts;
	cv::Mat tpl_desc, src_desc;
	detector->detectAndCompute(tpl, cv::noArray(), tpl_kpts, tpl_desc);
	detector->detectAndCompute(src, cv::noArray(), src_kpts, src_desc);
	printf("-- # template keypoints : %zd\n", tpl_kpts.size());
	printf("-- # source keypoints : %zd\n", src_kpts.size());

	if (selectedCommand == CMD_DETECT) { // Detect keypoints and display them, without matching
		cv::Mat tpl_out, src_out, out;
		cv::drawKeypoints(tpl, tpl_kpts, tpl_out);
		cv::drawKeypoints(src, src_kpts, src_out);
		if (tpl_out.rows < src_out.rows)
			tpl_out.resize(src_out.rows, cv::Scalar(0));
		else
			src_out.resize(tpl_out.rows, cv::Scalar(0));
		cv::hconcat(std::vector<cv::Mat>{ tpl_out, src_out }, out);
		cv::imshow("Feature Based Template Matching", out);
		return;
	}

	//-- Step 2: Match descriptors
	cv::Ptr<cv::DescriptorMatcher> matcher;
	switch (selectedMatcher) {
	case MATCHER_FLANN: matcher = cv::FlannBasedMatcher::create("FlannBased"); break;
	case MATCHER_BF: matcher = cv::BFMatcher::create("BruteForce"); break;
	default: printf("Unsupported choice of matcher(%d), exiting...\n", selectedMatcher); exit(1);
	}
	std::vector<cv::DMatch> matches;
	matcher->match(tpl_desc, src_desc, matches);
	printf("-- # matches : %zd\n", matches.size());

	//-- Step 3: Find "good" matches
	double min_dist = 100000000.0;
	for (int i = 0; i < tpl_desc.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
	}
	printf("-- Min dist : %f\n", min_dist);

	std::vector<cv::DMatch> good_matches;
	for (cv::DMatch match : matches)
		if (match.distance <= cv::max(5 * min_dist, 0.02))
			good_matches.push_back(match);
	printf("-- # good matches : %zd\n", good_matches.size());

	//-- Step 4: Draw "good" matches
	cv::Mat out;
	cv::drawMatches(tpl, tpl_kpts, src, src_kpts, good_matches, out,
		cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imshow("Feature Based Template Matching", out);

	for (cv::DMatch match : good_matches)
		printf("-- Keypoint 1: %d  -- Keypoint 2: %d -- dist: %f\n", match.queryIdx, match.trainIdx, match.distance);
}

void redraw()
{
	if (selectedMatcher != prevMatcher && selectedCommand == CMD_DETECT) {
		// changing matching method when matching isn't done shouldn't result in redrawing
		// but will remember the selected matching method
		prevMatcher = selectedMatcher;
		return;
	}
	if (selectedDetector != prevDetector
		|| selectedMatcher != prevMatcher
		|| selectedCommand != prevCommand)
		draw();
}

void detectorBtnCB(int state, void* userdata)
{
	selectedDetector = *(enum DetectorType*)userdata;
	redraw();
}

void matcherBtnCB(int state, void* userdata)
{
	selectedMatcher = *(enum MatcherType *)userdata;
	redraw();
}

void commandBtnCB(int state, void* userdata)
{
	selectedCommand = *(enum CommandType *)userdata;
	redraw();
}

int main()
{
	tpl = cv::imread(TEMPLATE_PATH, CV_LOAD_IMAGE_GRAYSCALE);
	src = cv::imread(SOURCE_PATH, CV_LOAD_IMAGE_GRAYSCALE);

	cv::namedWindow("Feature Based Template Matching", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

	const enum DetectorType detector[DETECTOR_SIZE] = { DETECTOR_SIFT, DETECTOR_SURF, DETECTOR_KAZE, DETECTOR_AKAZE, DETECTOR_ORB };
	cv::createButton(detector_name[DETECTOR_SIFT], detectorBtnCB, (void *)&detector[DETECTOR_SIFT]);
	cv::createButton(detector_name[DETECTOR_SURF], detectorBtnCB, (void *)&detector[DETECTOR_SURF]);
	cv::createButton(detector_name[DETECTOR_KAZE], detectorBtnCB, (void *)&detector[DETECTOR_KAZE]);
	cv::createButton(detector_name[DETECTOR_AKAZE], detectorBtnCB, (void *)&detector[DETECTOR_AKAZE]);
	cv::createButton(detector_name[DETECTOR_ORB], detectorBtnCB, (void *)&detector[DETECTOR_ORB]);

	const enum MatcherType matcher[MATCHER_SIZE] = { MATCHER_FLANN, MATCHER_BF };
	cv::createButton(matcher_name[MATCHER_FLANN], matcherBtnCB, (void *)&matcher[MATCHER_FLANN]);
	cv::createButton(matcher_name[MATCHER_BF], matcherBtnCB, (void *)&matcher[MATCHER_BF]);

	const enum CommandType command[CMD_SIZE] = { CMD_DETECT, CMD_DETECT_AND_MATCH };
	cv::createButton(command_name[CMD_DETECT], commandBtnCB, (void *)&command[CMD_DETECT]);
	cv::createButton(command_name[CMD_DETECT_AND_MATCH], commandBtnCB, (void *)&command[CMD_DETECT_AND_MATCH]);

	draw();
	cv::waitKey();

	return 0;
}