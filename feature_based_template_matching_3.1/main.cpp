#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>		// FlannBasedMatcher, KAZE, AKAZE
#include <opencv2/xfeatures2d.hpp>		// SIFT, SURF

#include "main.h"

#define TEMPLATE_PATH           // write template path here
#define SOURCE_PATH             // write source path here

#define WINDOW_NAME             "Feature Based Template Matching"

cv::Mat tpl;
cv::Mat src;

// Initial Settings
enum DetectorType selectedDetector = DETECTOR_SIFT, prevDetector;
enum MatcherType selectedMatcher = MATCHER_BF, prevMatcher;
enum CommandType selectedCommand = CMD_DETECT_AND_MATCH, prevCommand;

// Stringified names for enumerations for convenience
const std::string detector_name[DETECTOR_SIZE] = { "SIFT", "SURF", "KAZE", "AKAZE", "ORB", "BRISK", "AGAST", "FAST", "GFTT" };
const std::string matcher_name[MATCHER_SIZE] = { "FLANN", "Brute Force" };
const std::string command_name[CMD_SIZE] = { "Detect", "Detect and match" };

void update_status()
{
	prevDetector = selectedDetector;
	prevMatcher = selectedMatcher;
	prevCommand = selectedCommand;
}

void restore_status()
{
	selectedDetector = prevDetector;
	selectedMatcher = prevMatcher;
	selectedCommand = prevCommand;
}

void draw()
{
	printf("\n\n------------------------------ REDRAW ------------------------------\n");
	printf("-- Selected feature detecting method : %s\n", detector_name[selectedDetector].c_str());
	
	cv::Ptr<cv::Feature2D> detector;
	switch (selectedDetector) {
	case DETECTOR_SIFT: detector = cv::xfeatures2d::SiftFeatureDetector::create(); break;
	case DETECTOR_SURF: detector = cv::xfeatures2d::SurfFeatureDetector::create(); break;
	case DETECTOR_KAZE: detector = cv::KAZE::create(); break;
	case DETECTOR_AKAZE: detector = cv::AKAZE::create(); break;
	case DETECTOR_ORB: detector = cv::ORB::create(); break;
	case DETECTOR_BRISK: detector = cv::BRISK::create(); break;
	case DETECTOR_AGAST: detector = cv::AgastFeatureDetector::create(); break;
	case DETECTOR_FAST: detector = cv::FastFeatureDetector::create(); break;
	case DETECTOR_GFTT: detector = cv::GFTTDetector::create(); break;
	default: printf("Unsupported choice of feature detector(%d), exiting...\n", selectedDetector); exit(1);
	}

	//-- Step 1: Detect keypoints and compute descriptors
	std::vector<cv::KeyPoint> tpl_kpts, src_kpts;
	cv::Mat tpl_desc, src_desc;
	try {
		detector->detectAndCompute(tpl, cv::noArray(), tpl_kpts, tpl_desc);
		detector->detectAndCompute(src, cv::noArray(), src_kpts, src_desc);
	} catch (cv::Exception& exc) {
		char buf[100];
		snprintf(buf, 100, "Error: Selected detector(%s) is not implemented\n", detector_name[selectedDetector].c_str());
		cv::displayStatusBar(WINDOW_NAME, buf);
		restore_status();
		return;
	}

	char buf[100];
	if (selectedCommand != CMD_DETECT) {
		printf("-- Selected matching method : %s\n", matcher_name[selectedMatcher].c_str());
		printf("-- Selected command : %s\n", command_name[selectedCommand].c_str());

		snprintf(buf, 100, "%s / %s\n", command_name[selectedCommand].c_str(), detector_name[selectedDetector].c_str());
		cv::displayStatusBar(WINDOW_NAME, cv::String(buf));
	}
	snprintf(buf, 100, "%s / %s / %s\n", command_name[selectedCommand].c_str(), detector_name[selectedDetector].c_str(), matcher_name[selectedMatcher].c_str());
	cv::displayStatusBar(WINDOW_NAME, cv::String(buf));

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
		cv::imshow(WINDOW_NAME, out);
		update_status();
		return;
	}

	//-- Step 2: Match descriptors
	cv::Ptr<cv::DescriptorMatcher> matcher;
	switch (selectedMatcher) {
	case MATCHER_FLANN: matcher = cv::FlannBasedMatcher::create("FlannBased"); break;
	case MATCHER_BF: matcher = cv::BFMatcher::create("BruteForce"); break;
	default: printf("Unsupported choice of matcher(%d), exiting...\n", selectedMatcher); exit(1);
	}

// 0: use match, find best matches between template and source
// 1: use knnMatch, to see if not-best matches has potential to be used to match recurrent objects
#if 0
	std::vector<std::vector<cv::DMatch>> knnMatches;
	matcher->knnMatch(tpl_desc, src_desc, knnMatches, 20);
	double min_dist = 100000000.0;
	for (int i = 0; i < tpl_desc.rows; i++) {
		double dist = knnMatches[i][0].distance;
		if (dist < min_dist)
			min_dist = dist;
	}
	printf("-- Min dist : %f\n", min_dist);

	std::vector<std::vector<cv::DMatch>> matchesDisplayed;
	int i = 0;
	for (std::vector<cv::DMatch> matchesKpt : knnMatches) {
		cv::DMatch match = matchesKpt[0];
		if (match.distance > cv::max(5 * min_dist, 0.02))
			continue;
		std::vector<cv::DMatch> *temp = new std::vector<cv::DMatch>;
		matchesDisplayed.push_back(*temp);
		for (cv::DMatch match : matchesKpt) {
			if (match.distance <= cv::max(13 * min_dist, 0.02))
				matchesDisplayed[i].push_back(match);
		}
		i++;
	}
	printf("-- # good matches : %zd\n", matchesDisplayed.size());

	std::vector<std::vector<cv::DMatch>> goodMatchesDisplayed;
	for (int i = 0; i < matchesDisplayed.size(); i++) {
		printf("-- # mathces for %i : %zd\n", i, matchesDisplayed[i].size());
		if (matchesDisplayed[i].size() > 9)
			goodMatchesDisplayed.push_back(matchesDisplayed[i]);
	}
	cv::Mat out;
	cv::drawMatches(tpl, tpl_kpts, src, src_kpts, goodMatchesDisplayed, out,
		cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<std::vector<char>>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imshow(WINDOW_NAME, out);
	cv::imwrite("C:/Users/lee.namgoo/Desktop/screenshots_mlcc/out.png", out);
#else
	std::vector<cv::DMatch> matches;
	try {
		matcher->match(tpl_desc, src_desc, matches);
	} catch (cv::Exception& exc) {
		char buf[100];
		snprintf(buf, 100, "Error: Selected matcher(%s) is not applicable\n", matcher_name[selectedMatcher].c_str());
		cv::displayStatusBar(WINDOW_NAME, buf);
		restore_status();
		return;
	}
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
	cv::imshow(WINDOW_NAME, out);

	for (cv::DMatch match : good_matches)
		printf("-- Keypoint 1: %d  -- Keypoint 2: %d -- dist: %f\n", match.queryIdx, match.trainIdx, match.distance);
#endif

	update_status();
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

	cv::GaussianBlur(tpl, tpl, cv::Size(3, 3), 0.0);
	cv::GaussianBlur(src, src, cv::Size(3, 3), 0.0);

	cv::namedWindow(WINDOW_NAME, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

	const enum DetectorType detector_holder[DETECTOR_SIZE] = { DETECTOR_SIFT, DETECTOR_SURF, DETECTOR_KAZE, DETECTOR_AKAZE, DETECTOR_ORB, DETECTOR_BRISK, DETECTOR_AGAST, DETECTOR_FAST, DETECTOR_GFTT };
	cv::createButton(detector_name[DETECTOR_SIFT], detectorBtnCB, (void *)&detector_holder[DETECTOR_SIFT]);
	cv::createButton(detector_name[DETECTOR_SURF], detectorBtnCB, (void *)&detector_holder[DETECTOR_SURF]);
	cv::createButton(detector_name[DETECTOR_KAZE], detectorBtnCB, (void *)&detector_holder[DETECTOR_KAZE]);
	cv::createButton(detector_name[DETECTOR_AKAZE], detectorBtnCB, (void *)&detector_holder[DETECTOR_AKAZE]);
	cv::createButton(detector_name[DETECTOR_ORB], detectorBtnCB, (void *)&detector_holder[DETECTOR_ORB]);
	cv::createButton(detector_name[DETECTOR_BRISK], detectorBtnCB, (void *)&detector_holder[DETECTOR_BRISK]);
	cv::createButton(detector_name[DETECTOR_AGAST], detectorBtnCB, (void *)&detector_holder[DETECTOR_AGAST]);
	cv::createButton(detector_name[DETECTOR_FAST], detectorBtnCB, (void *)&detector_holder[DETECTOR_FAST]);
	cv::createButton(detector_name[DETECTOR_GFTT], detectorBtnCB, (void *)&detector_holder[DETECTOR_GFTT]);

	int dummy = 0;
	cv::createTrackbar("Sep1", cv::String(), &dummy, 100);

	const enum MatcherType matcher_holder[MATCHER_SIZE] = { MATCHER_FLANN, MATCHER_BF };
	cv::createButton(matcher_name[MATCHER_FLANN], matcherBtnCB, (void *)&matcher_holder[MATCHER_FLANN]);
	cv::createButton(matcher_name[MATCHER_BF], matcherBtnCB, (void *)&matcher_holder[MATCHER_BF]);

	cv::createTrackbar("Sep2", cv::String(), &dummy, 100);

	const enum CommandType command_holder[CMD_SIZE] = { CMD_DETECT, CMD_DETECT_AND_MATCH };
	cv::createButton(command_name[CMD_DETECT], commandBtnCB, (void *)&command_holder[CMD_DETECT]);
	cv::createButton(command_name[CMD_DETECT_AND_MATCH], commandBtnCB, (void *)&command_holder[CMD_DETECT_AND_MATCH]);

	draw();

	cv::waitKey();

	return 0;
}
