#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <sstream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <wingdi.h>
#elif __APPLE__
#include <ApplicationServices/ApplicationServices.h>
#elif __linux__
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

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

    #ifdef _WIN32
    static HBITMAP CaptureScreen(int x = 0, int y = 0, int width = GetSystemMetrics(SM_CXSCREEN), int height = GetSystemMetrics(SM_CYSCREEN)) {
        HDC hScreenDC = GetDC(NULL);
        HDC hMemoryDC = CreateCompatibleDC(hScreenDC);

        HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, width, height);
        HBITMAP hOldBitmap = (HBITMAP)SelectObject(hMemoryDC, hBitmap);

        BitBlt(hMemoryDC, 0, 0, width, height, hScreenDC, x, y, SRCCOPY);

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

    static HWND FindWindowByTitle(const std::wstring& title) {
        return FindWindow(NULL, title.c_str());
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

    cv::Mat ResourceToMat(int resourceID, const std::string& resourceType) {
        HRSRC hResource = FindResource(NULL, MAKEINTRESOURCE(resourceID), resourceType.c_str());
        if (hResource == NULL) {
            return cv::Mat();
        }

        HGLOBAL hLoadedResource = LoadResource(NULL, hResource);
        if (hLoadedResource == NULL) {
            return cv::Mat();
        }

        DWORD dwResourceSize = SizeofResource(NULL, hResource);
        const void* pResourceData = LockResource(hLoadedResource);

        std::vector<uchar> buffer((uchar*)pResourceData, (uchar*)pResourceData + dwResourceSize);
        cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        return image;
    }

    #elif __APPLE__
    CGImageRef CaptureScreen(int x = 0, int y = 0, int width = CGDisplayPixelsWide(kCGDirectMainDisplay), int height = CGDisplayPixelsHigh(kCGDirectMainDisplay)) {
        CGRect captureRect = CGRectMake(x, y, width, height);
        return CGWindowListCreateImage(captureRect, kCGWindowListOptionOnScreenOnly, kCGNullWindowID, kCGWindowImageDefault);
    }

    CGImageRef CaptureWindow(CGWindowID windowID) {
        CGImageRef image = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionOnScreenOnly,
            windowID,
            kCGWindowImageDefault
        );
        
        if (!image) {
            throw std::runtime_error("Failed to capture window.");
        }
        return image;
    }

    static WindowID FindWindowByTitle(const std::string& title) {
        uint32_t windowListSize;
        CGWindowID *windowList = NULL;

        windowList = CGWindowListCopyWindowIDs(kCGWindowListOptionAll, &windowListSize);
        for (uint32_t i = 0; i < windowListSize; ++i) {
            NSDictionary *info = (NSDictionary *)CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, windowList[i]);
            if (info) {
                std::string windowName = (const char *)CFStringGetCStringPtr((CFStringRef)info[kCGWindowName], kCFStringEncodingUTF8);
                if (windowName == title) {
                    return windowList[i];
                }
            }
        }
        return 0;
    }

    static cv::Mat CGImageToMat(CGImageRef image) {
        size_t width = CGImageGetWidth(image);
        size_t height = CGImageGetHeight(image);
        
        cv::Mat mat(height, width, CV_8UC4);
        
        CGContextRef context = CGBitmapContextCreate(mat.data, width, height, 8, mat.step[0], 
                                                    CGColorSpaceCreateDeviceRGB(), 
                                                    kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        if (!context) {
            throw std::runtime_error("Failed to create CGContext.");
        }
        
        CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
        CGContextRelease(context);
        
        return mat;
    }

    static void ClickAtPosition(int x, int y) {
        CGPoint point = CGPointMake(x, y);
        CGEventRef downEvent = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseDown, point, kCGEventSourceStateHIDSystemState);
        CGEventRef upEvent = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseUp, point, kCGEventSourceStateHIDSystemState);

        CGEventPost(kCGHIDEventTap, downEvent);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CGEventPost(kCGHIDEventTap, upEvent);

        CFRelease(downEvent);
        CFRelease(upEvent);
    }

    #elif __linux__
    XImage* CaptureScreen(Display* display, int x = 0, int y = 0, int width = 0, int height = 0) {
        Window root = DefaultRootWindow(display);
        XWindowAttributes attributes;
        XGetWindowAttributes(display, root, &attributes);

        if (width == 0) width = attributes.width;
        if (height == 0) height = attributes.height;

        return XGetImage(display, root, x, y, width, height, AllPlanes, ZPixmap);
    }

    Pixmap CaptureWindow(Display* display, Window window) {
        XWindowAttributes attrs;
        XGetWindowAttributes(display, window, &attrs);

        Pixmap pixmap = XCreatePixmap(display, window, attrs.width, attrs.height, attrs.depth);
        if (!pixmap) {
            throw std::runtime_error("Failed to create pixmap.");
        }

        GC gc = XCreateGC(display, pixmap, 0, NULL);
        XCopyArea(display, window, pixmap, gc, 0, 0, attrs.width, attrs.height, 0, 0);
        XFreeGC(display, gc);

        return pixmap;
    }

    static Window FindWindowByTitle(Display* display, const std::string& title) {
        Window root = DefaultRootWindow(display);
        Window returnedRoot, returnedParent;
        Window* children;
        unsigned int numChildren;

        if (XQueryTree(display, root, &returnedRoot, &returnedParent, &children, &numChildren)) {
            for (unsigned int i = 0; i < numChildren; ++i) {
                char* windowTitle;
                Atom name = XInternAtom(display, "WM_NAME", True);
                if (XGetWindowProperty(display, children[i], name, 0, 1024, False, AnyPropertyType, 
                                    &name, &format, &items, &bytes, (unsigned char**)&windowTitle) == Success) {
                    if (windowTitle && title == windowTitle) {
                        XFree(windowTitle);
                        return children[i];
                    }
                    XFree(windowTitle);
                }
            }
        }
        return 0;
    }

    static cv::Mat XImageToMat(XImage* xImage) {
        int width = xImage->width;
        int height = xImage->height;

        cv::Mat mat(height, width, CV_8UC4);

        memcpy(mat.data, xImage->data, height * xImage->bytes_per_line);

        return mat;
    }

    static void ClickAtPosition(int x, int y) {
        Display* display = XOpenDisplay(NULL);
        if (!display) {
            std::cerr << "Cannot open display!" << std::endl;
            return;
        }

        XWarpPointer(display, None, DefaultRootWindow(display), 0, 0, 0, 0, x, y);
        XFlush(display);

        XEvent event;
        event.xbutton.type = ButtonPress;
        event.xbutton.button = Button1;
        event.xbutton.root = DefaultRootWindow(display);
        event.xbutton.subwindow = DefaultRootWindow(display);
        event.xbutton.x = x;
        event.xbutton.y = y;
        event.xbutton.x_root = x;
        event.xbutton.y_root = y;
        event.xbutton.same_screen = True;

        XSendEvent(display, PointerWindow, True, ButtonPressMask, &event);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        event.xbutton.type = ButtonRelease;
        XSendEvent(display, PointerWindow, True, ButtonReleaseMask, &event);
        XFlush(display);

        XCloseDisplay(display);
    }
    #endif
    
    cv::Mat ByteArrayToMat(const std::vector<uchar>& byteArray) {
        cv::Mat image = cv::imdecode(byteArray, cv::IMREAD_COLOR);
        return image;
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