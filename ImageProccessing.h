#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <windows.h>
#include <filesystem>
#include <sstream>
#include <atlbase.h>
#include <algorithm>

class IP {
public:
    static cv::Mat rotateImage(const cv::Mat& image, const std::string& direction, double angle) {
        cv::Point2f center(image.cols / 2.0, image.rows / 2.0);

        if (direction == "left") {
            angle = -angle;
        }
        else if (direction != "right") {
            std::cerr << "Invalid direction. Use 'left' or 'right'." << std::endl;
            return image;
        }

        cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

        cv::Mat rotatedImage;

        cv::warpAffine(image, rotatedImage, rotationMatrix, image.size());

        return rotatedImage;
    }

    static cv::Rect findImageInImage(const cv::Mat& largeImage, const cv::Mat& smallImage, double scale = 1.0, bool grayscale = false) {
        if (scale <= 0.0 || scale > 1.0) {
            throw std::invalid_argument("Scale must be between 0 and 1.");
        }

        cv::Mat largeCopy, smallCopy;

        if (scale != 1.0) {
            cv::resize(largeImage, largeCopy, cv::Size(), scale, scale);
            cv::resize(smallImage, smallCopy, cv::Size(), scale, scale);
        }
        else {
            largeCopy = largeImage;
            smallCopy = smallImage;
        }

        if (grayscale) {
            cv::cvtColor(largeCopy, largeCopy, cv::COLOR_BGR2GRAY);
            cv::cvtColor(smallCopy, smallCopy, cv::COLOR_BGR2GRAY);
        }

        cv::Mat result;
        cv::matchTemplate(largeCopy, smallCopy, result, cv::TM_CCOEFF_NORMED);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        cv::Rect matchRect(maxLoc.x, maxLoc.y, smallCopy.cols, smallCopy.rows);

        int x = static_cast<int>(matchRect.x / scale);
        int y = static_cast<int>(matchRect.y / scale);
        int width = static_cast<int>(matchRect.width / scale);
        int height = static_cast<int>(matchRect.height / scale);

        x = (x < 0) ? 0 : (x + width > largeImage.cols) ? largeImage.cols - width : x;
        y = (y < 0) ? 0 : (y + height > largeImage.rows) ? largeImage.rows - height : y;

        cv::Rect roi(x, y, width, height);
        return roi;
    }

    static cv::Rect findImageInImageORB(const cv::Mat& largeImage, const cv::Mat& smallImage, int minMatchScore = 230, double scale = 1.0, bool debug = false) {
        if (scale <= 0.0 || scale > 1.0) {
            throw std::invalid_argument("Scale must be between 0 and 1.");
        }
        
        minMatchScore = std::clamp(minMatchScore, 0, 256);

        cv::Mat largeCopy, smallCopy;

        if (scale != 1.0) {
            cv::resize(largeImage, largeCopy, cv::Size(), scale, scale);
            cv::resize(smallImage, smallCopy, cv::Size(), scale, scale);
        }
        else {
            largeCopy = largeImage;
            smallCopy = smallImage;
        }

        std::vector<cv::KeyPoint> keypointsLarge, keypointsSmall;
        cv::Mat descriptorsLarge, descriptorsSmall;

        computeKeypointsAndDescriptors(largeCopy, keypointsLarge, descriptorsLarge);
        computeKeypointsAndDescriptors(smallCopy, keypointsSmall, descriptorsSmall);

        if (descriptorsLarge.empty() || descriptorsSmall.empty()) {
            std::cerr << "Error: One or both images failed to produce descriptors.\n";
            return cv::Rect(0, 0, 0, 0);
        }

        if (debug) {
            cv::Mat largeKeypointsImg, smallKeypointsImg;
            cv::drawKeypoints(largeCopy, keypointsLarge, largeKeypointsImg, cv::Scalar(0, 255, 0));
            cv::drawKeypoints(smallCopy, keypointsSmall, smallKeypointsImg, cv::Scalar(0, 255, 0));

            cv::imshow("Large Image Keypoints", largeKeypointsImg);
            cv::imshow("Small Image Keypoints", smallKeypointsImg);
        }

        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptorsSmall, descriptorsLarge, matches);

        if (matches.empty()) {
            std::cerr << "Error: No matches found between descriptors.\n";
            return cv::Rect(0, 0, 0, 0);
        }
        
        float minDistance = std::min_element(matches.begin(), matches.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance;
            })->distance;
        
        float maxAcceptableDistance = minDistance + (minMatchScore / 256.0f * 256.0f);
        std::vector<cv::DMatch> goodMatches;
        std::copy_if(matches.begin(), matches.end(), std::back_inserter(goodMatches),
            [maxAcceptableDistance](const cv::DMatch& m) {
                return m.distance <= maxAcceptableDistance;
            });

        if (debug) {
            cv::Mat matchImg;
            cv::drawMatches(smallCopy, keypointsSmall, largeCopy, keypointsLarge, goodMatches, matchImg);

            cv::imshow("Matches", matchImg);
            cv::waitKey(0); 
        }

        
        if (goodMatches.size() < 4) {
            std::cerr << "Error: Not enough good matches found to compute homography.\n";
            return cv::Rect(0, 0, 0, 0);
        }
        
        std::vector<cv::Point2f> pointsSmall, pointsLarge;
        for (const auto& match : goodMatches) {
            pointsSmall.push_back(keypointsSmall[match.queryIdx].pt);
            pointsLarge.push_back(keypointsLarge[match.trainIdx].pt);
        }
        
        cv::Mat homography = cv::findHomography(pointsSmall, pointsLarge, cv::RANSAC);
        
        if (homography.empty()) {
            std::cerr << "Error: Homography computation failed.\n";
            return cv::Rect(0, 0, 0, 0);
        }
        
        std::vector<cv::Point2f> smallCorners = {
            cv::Point2f(0, 0),
            cv::Point2f(static_cast<float>(smallCopy.cols), 0),
            cv::Point2f(static_cast<float>(smallCopy.cols), static_cast<float>(smallCopy.rows)),
            cv::Point2f(0, static_cast<float>(smallCopy.rows))
        };

        std::vector<cv::Point2f> largeCorners(4);
        cv::perspectiveTransform(smallCorners, largeCorners, homography);

        cv::Rect boundingRect = cv::boundingRect(largeCorners);
        if (!isAspectRatioClose(boundingRect, smallImage, 0.2)) {
            return cv::Rect(0, 0, 0, 0);
        }
        return boundingRect;
    }

    static cv::Rect findImageInImageORB(const cv::Mat& largeImage, const cv::Mat& smallImage,
        const std::vector<cv::KeyPoint>& keypointsLarge, const cv::Mat& descriptorsLarge,
        const std::vector<cv::KeyPoint>& keypointsSmall, const cv::Mat& descriptorsSmall,
        int minMatchScore = 230, bool debug = false) {
        
        minMatchScore = std::clamp(minMatchScore, 0, 256);

        if (descriptorsLarge.empty() || descriptorsSmall.empty()) {
            std::cerr << "Error: One or both sets of descriptors are empty.\n";
            return cv::Rect(0, 0, 0, 0);
        }

        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptorsSmall, descriptorsLarge, matches);

        if (matches.empty()) {
            std::cerr << "Error: No matches found between descriptors.\n";
            return cv::Rect(0, 0, 0, 0);
        }

        float minDistance = std::min_element(matches.begin(), matches.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance;
            })->distance;

        float maxAcceptableDistance = minDistance + (minMatchScore / 256.0f * 256.0f);
        std::vector<cv::DMatch> goodMatches;
        std::copy_if(matches.begin(), matches.end(), std::back_inserter(goodMatches),
            [maxAcceptableDistance](const cv::DMatch& m) {
                return m.distance <= maxAcceptableDistance;
            });

        if (debug) {
            cv::Mat matchImg;
            cv::drawMatches(smallImage, keypointsSmall, largeImage, keypointsLarge, goodMatches, matchImg);

            cv::imshow("Matches", matchImg);
            cv::waitKey(0); 
        }

        if (goodMatches.size() < 4) {
            std::cerr << "Error: Not enough good matches found to compute homography.\n";
            return cv::Rect(0, 0, 0, 0);
        }

        std::vector<cv::Point2f> pointsSmall, pointsLarge;
        for (const auto& match : goodMatches) {
            pointsSmall.push_back(keypointsSmall[match.queryIdx].pt);
            pointsLarge.push_back(keypointsLarge[match.trainIdx].pt);
        }

        cv::Mat homography = cv::findHomography(pointsSmall, pointsLarge, cv::RANSAC);

        if (homography.empty()) {
            std::cerr << "Error: Homography computation failed.\n";
            return cv::Rect(0, 0, 0, 0);
        }

        std::vector<cv::Point2f> smallCorners = {
            cv::Point2f(0, 0),
            cv::Point2f(static_cast<float>(smallImage.cols), 0),
            cv::Point2f(static_cast<float>(smallImage.cols), static_cast<float>(smallImage.rows)),
            cv::Point2f(0, static_cast<float>(smallImage.rows))
        };

        std::vector<cv::Point2f> largeCorners(4);
        cv::perspectiveTransform(smallCorners, largeCorners, homography);

        cv::Rect boundingRect = cv::boundingRect(largeCorners);

        if (boundingRect.width <= 0 || boundingRect.height <= 0 ||
            boundingRect.x < 0 || boundingRect.y < 0 ||
            boundingRect.x + boundingRect.width > largeImage.cols ||
            boundingRect.y + boundingRect.height > largeImage.rows) {
            return cv::Rect(0, 0, 0, 0);
        }
        if (!isAspectRatioClose(boundingRect, smallImage, 0.2)) {
            return cv::Rect(0, 0, 0, 0);
        }
        return boundingRect;
    }

    static void computeKeypointsAndDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
        int imageArea = image.cols * image.rows;

        int limit = std::clamp(static_cast<int>(imageArea * 0.005), 500, INT_MAX);

        cv::Ptr<cv::ORB> orb = cv::ORB::create(limit);

        orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    }

    static cv::Mat convertToGrayScale(const cv::Mat& inputImage) {
        if (inputImage.empty()) {
            std::cerr << "Error: Input image is empty.\n";
            return inputImage;
        }

        cv::Mat grayImage;
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

        return grayImage;
    }

    static HBITMAP CaptureScreen() {
        HDC hScreenDC = GetDC(NULL);
        
        HDC hMemoryDC = CreateCompatibleDC(hScreenDC);

        int width = GetSystemMetrics(SM_CXSCREEN);
        int height = GetSystemMetrics(SM_CYSCREEN);

        HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, width, height);

        HBITMAP hOldBitmap = (HBITMAP)SelectObject(hMemoryDC, hBitmap);

        BitBlt(hMemoryDC, 0, 0, width, height, hScreenDC, 0, 0, SRCCOPY);

        SelectObject(hMemoryDC, hOldBitmap);

        DeleteDC(hMemoryDC);
        ReleaseDC(NULL, hScreenDC);

        return hBitmap;
    }

    static HBITMAP CaptureWindow(HWND hwnd) {
        HDC hWindowDC = GetWindowDC(hwnd);
        RECT rc;
        GetWindowRect(hwnd, &rc);
        int width = rc.right - rc.left;
        int height = rc.bottom - rc.top;

        HDC hMemoryDC = CreateCompatibleDC(hWindowDC);
        HBITMAP hBitmap = CreateCompatibleBitmap(hWindowDC, width, height);
        HBITMAP hOldBitmap = (HBITMAP)SelectObject(hMemoryDC, hBitmap);

        if (!PrintWindow(hwnd, hMemoryDC, PW_RENDERFULLCONTENT)) {
            BitBlt(hMemoryDC, 0, 0, width, height, hWindowDC, 0, 0, SRCCOPY);
        }

        SelectObject(hMemoryDC, hOldBitmap);
        DeleteDC(hMemoryDC);
        ReleaseDC(hwnd, hWindowDC);

        return hBitmap;
    }

    static HBITMAP CropHBitmap(HBITMAP hBitmap, const cv::Rect& cropRect) {
        BITMAP bitmap;
        GetObject(hBitmap, sizeof(BITMAP), &bitmap);

        if (cropRect.x < 0 || cropRect.y < 0 ||
            cropRect.x + cropRect.width > bitmap.bmWidth ||
            cropRect.y + cropRect.height > bitmap.bmHeight) {
            
            return nullptr;
        }

        HDC hWindowDC = GetDC(nullptr);
        HDC hMemoryDC = CreateCompatibleDC(hWindowDC);
        HBITMAP hNewBitmap = CreateCompatibleBitmap(hWindowDC, cropRect.width, cropRect.height);
        HBITMAP hOldBitmap = (HBITMAP)SelectObject(hMemoryDC, hNewBitmap);

        BitBlt(hMemoryDC, 0, 0, cropRect.width, cropRect.height, hWindowDC, cropRect.x, cropRect.y, SRCCOPY);

        SelectObject(hMemoryDC, hOldBitmap);
        DeleteDC(hMemoryDC);
        ReleaseDC(nullptr, hWindowDC);

        return hNewBitmap;
    }

    static cv::Mat HBitmapToMat(HBITMAP hBitmap) {
        BITMAP bmp;
        GetObject(hBitmap, sizeof(BITMAP), &bmp);
        int width = bmp.bmWidth;
        int height = bmp.bmHeight;

        HDC hMemoryDC = CreateCompatibleDC(NULL);
        HBITMAP hOldBitmap = (HBITMAP)SelectObject(hMemoryDC, hBitmap);

        cv::Mat mat(height, width, CV_8UC4); 

        BITMAPINFO bi = { 0 };
        bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth = width;
        bi.bmiHeader.biHeight = -height; 
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;

        GetDIBits(hMemoryDC, hBitmap, 0, height, mat.data, &bi, DIB_RGB_COLORS);

        SelectObject(hMemoryDC, hOldBitmap);
        DeleteDC(hMemoryDC);

        return mat;
    }

    static HWND FindWindowByTitle(const std::wstring& title) {
        return FindWindow(NULL, title.c_str());
    }

    static void ClickAtPosition(int x, int y) {
        INPUT inputs[2] = {};
        
        inputs[0].type = INPUT_MOUSE;
        inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        inputs[0].mi.dx = (LONG)x;
        inputs[0].mi.dy = (LONG)y;
        inputs[0].mi.dwExtraInfo = 0;

        inputs[1].type = INPUT_MOUSE;
        inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;
        inputs[1].mi.dx = (LONG)x;
        inputs[1].mi.dy = (LONG)y;
        inputs[1].mi.dwExtraInfo = 0;

        
        SetCursorPos(x, y);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        SendInput(2, inputs, sizeof(INPUT));
    }

    static void displayImage(const cv::Mat& image, const std::string& windowName) {
        if (image.empty()) {
            std::cerr << "Could not open or find the image!" << std::endl;
            return;
        }
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowName, image);
        cv::waitKey(0);
    }

    static cv::Mat getRegionOfInterest(const cv::Mat& image, const cv::Rect& roi) {
        if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
            roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
            std::cerr << "Invalid ROI!" << std::endl;
            return cv::Mat(); 
        }

        cv::Mat roiImage = image(roi);
        return roiImage;
    }

    static cv::Rect GetRoiFromHBitmap(const std::string& keyphrase, HBITMAP hBitmap) {
        
        auto parseFraction = [](const std::string& fractionStr) -> double {
            size_t pos = fractionStr.find('/');
            if (pos != std::string::npos) {
                double numerator = std::stod(fractionStr.substr(0, pos));
                double denominator = std::stod(fractionStr.substr(pos + 1));
                return numerator / denominator;
            }
            return std::stod(fractionStr);
            };

        
        BITMAP bitmap;
        GetObject(hBitmap, sizeof(BITMAP), &bitmap);
        cv::Size imageSize(bitmap.bmWidth, bitmap.bmHeight);

        if (keyphrase == "default") {
            return cv::Rect(0, 0, imageSize.width, imageSize.height);
        }

        std::istringstream ss(keyphrase);
        std::string direction;
        std::string fractionStr;

        
        cv::Rect roi(0, 0, imageSize.width, imageSize.height);

        while (ss >> direction >> fractionStr) {
            double fraction = parseFraction(fractionStr);
            int width = imageSize.width;
            int height = imageSize.height;

            if (direction == "right") {
                int newWidth = width * fraction;
                int newX = width - newWidth;
                roi = cv::Rect(newX, roi.y, newWidth, roi.height);
            }
            else if (direction == "left") {
                int newWidth = width * fraction;
                roi = cv::Rect(roi.x, roi.y, newWidth, roi.height);
            }
            else if (direction == "bottom") {
                int newHeight = height * fraction;
                int newY = height - newHeight;
                roi = cv::Rect(roi.x, newY, roi.width, newHeight);
            }
            else if (direction == "top") {
                int newHeight = height * fraction;
                roi = cv::Rect(roi.x, roi.y, roi.width, newHeight);
            }
            else if (direction == "center") {
                int newWidth = width * fraction;
                int newHeight = height * fraction;
                int newX = (width - newWidth) / 2;
                int newY = (height - newHeight) / 2;
                roi = cv::Rect(newX, newY, newWidth, newHeight);
            }
            else {
                std::cerr << "Invalid direction: " << direction << std::endl;
                return cv::Rect(0, 0, width, height);
            }
        }

        
        roi.x = (roi.x > 0) ? roi.x : 0;
        roi.y = (roi.y > 0) ? roi.y : 0;
        roi.width = (roi.width < imageSize.width - roi.x) ? roi.width : imageSize.width - roi.x;
        roi.height = (roi.height < imageSize.height - roi.y) ? roi.height : imageSize.height - roi.y;
        return roi;
    }

    static cv::Rect getRoiFromKeyphrase(const std::string& keyphrase, const cv::Size& imageSize) {
        
        auto parseFraction = [](const std::string& fractionStr) -> double {
            size_t pos = fractionStr.find('/');
            if (pos != std::string::npos) {
                double numerator = std::stod(fractionStr.substr(0, pos));
                double denominator = std::stod(fractionStr.substr(pos + 1));
                return numerator / denominator;
            }
            return std::stod(fractionStr);
            };

        if (keyphrase == "default") {
            return cv::Rect(0, 0, imageSize.width, imageSize.height);
        }

        std::istringstream ss(keyphrase);
        std::string direction;
        std::string fractionStr;

        cv::Rect roi(0, 0, imageSize.width, imageSize.height);

        while (ss >> direction >> fractionStr) {
            double fraction = parseFraction(fractionStr);
            int width = imageSize.width;
            int height = imageSize.height;

            if (direction == "right") {
                int newWidth = width * fraction;
                int newX = width - newWidth;
                roi = cv::Rect(newX, roi.y, newWidth, roi.height);
            }
            else if (direction == "left") {
                int newWidth = width * fraction;
                roi = cv::Rect(roi.x, roi.y, newWidth, roi.height);
            }
            else if (direction == "bottom") {
                int newHeight = height * fraction;
                int newY = height - newHeight;
                roi = cv::Rect(roi.x, newY, roi.width, newHeight);
            }
            else if (direction == "top") {
                int newHeight = height * fraction;
                roi = cv::Rect(roi.x, roi.y, roi.width, newHeight);
            }
            else if (direction == "center") {
                int newWidth = width * fraction;
                int newHeight = height * fraction;
                int newX = (width - newWidth) / 2;
                int newY = (height - newHeight) / 2;
                roi = cv::Rect(newX, newY, newWidth, newHeight);
            }
            else {
                std::cerr << "Invalid direction: " << direction << std::endl;
                return cv::Rect(0, 0, width, height);
            }
        }

        roi.x = (roi.x > 0) ? roi.x : 0;
        roi.y = (roi.y > 0) ? roi.y : 0;
        roi.width = (roi.width < imageSize.width - roi.x) ? roi.width : imageSize.width - roi.x;
        roi.height = (roi.height < imageSize.height - roi.y) ? roi.height : imageSize.height - roi.y;

        return roi;
    }
    
private:
    static bool isAspectRatioClose(const cv::Rect& rect, const cv::Mat& smallImage, double tolerance = 0.1) {
        double rectAspectRatio = static_cast<double>(rect.width) / rect.height;
        double rectRotatedAspectRatio = static_cast<double>(rect.height) / rect.width;
        double smallImageAspectRatio = static_cast<double>(smallImage.cols) / smallImage.rows;

        return (std::abs(rectAspectRatio - smallImageAspectRatio) <= tolerance) ||
            (std::abs(rectRotatedAspectRatio - smallImageAspectRatio) <= tolerance);
    }
};