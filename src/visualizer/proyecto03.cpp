
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include "helper/bounding_box.h"
#include "tracker/tracker.h"
#include "network/regressor.h"

using namespace std;
 
// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

ifstream inputFile;	
ofstream outFile;

char fileName[100];
char outFileName[100];
char outLine[200];

int lastX0 = 0;
int lastY0 = 0;
int lastW = 0;
int lastH = 0;
int lastRealIdx = 0;

cv::Rect2d* GetGroundBoxFromFile(int idxBtfly, int idxCam, int idxFrame, int rows, int cols) {

	string line;	
	int lineIdx;

	cv::Rect2d *bboxReturn;
	int x0s[4];
	int y0s[4];
	int ws[4];
	int hs[4];
	double distances[4];

	switch(idxCam) {
		case 0:
			sprintf(fileName, "butterflies_videos_annotations/01-NoBlur_4Butterflies/Camera0/1-%04d.txt", idxFrame);
			break;
		case 1:
			sprintf(fileName, "butterflies_videos_annotations/01-NoBlur_4Butterflies/Camera1/2-%04d.txt", idxFrame);
			break;
		case 2:
			sprintf(fileName, "butterflies_videos_annotations/01-NoBlur_4Butterflies/Camera2/3-%04d.txt", idxFrame);
			break;
		default:
			sprintf(fileName, "butterflies_videos_annotations/01-NoBlur_4Butterflies/Camera0/1-%04d.txt", idxFrame);
			break;
	}//switch(camIdx)  ends	

	//cout << fileName << '\n';

	inputFile.open(fileName);

	if (inputFile.is_open()) {
		lineIdx = 0;
    	while ( getline (inputFile, line) ) {
		//getline(inputFile, line);
      		//cout << line << '\n';	

			vector<string> tokens;
			stringstream check1(line);
			string intermediate;

			// Tokenizing w.r.t. space ' ' 
    		while(getline(check1, intermediate, ' ')) 
        		tokens.push_back(intermediate);	

			string x0Str = tokens[1];
			string y0Str = tokens[2];
			string wStr = tokens[3];
			string hStr = tokens[4];	

			x0s[lineIdx] = (int)(atof(x0Str.c_str()) * cols);
			y0s[lineIdx] = (int)(atof(y0Str.c_str()) * rows);
			ws[lineIdx] = (int)(atof(wStr.c_str()) * cols);
			hs[lineIdx] = (int)(atof(hStr.c_str()) * rows);	
			cv::Point a(x0s[lineIdx], y0s[lineIdx]);
			cv::Point b(lastX0, lastY0);
			distances[lineIdx] = cv::norm(a - b);	

			++lineIdx;
    	}//while ( getline (inputFile, line) )  ends

		//Closing file
    	inputFile.close();	

		double minDistance = distances[0];
		double distThres = 80.0;
		int realIdx = 0;
		bool btflyGone = false;
		int minThresCnt = 0;
			
		for(int realIdxTmp = 0; realIdxTmp < lineIdx; realIdxTmp++) {

			if(minDistance <= distThres)
				++minThresCnt;

			if(minDistance > distances[realIdxTmp]) {
				minDistance = distances[realIdxTmp];
				realIdx = realIdxTmp;
			}//if(minDistance > distances[realIdx]) ends

		}//for(int realIdxTmp = 0; realIdxTmp < lineIdx; realIdxTmp++) ends

		if(idxFrame == 1) {
			realIdx = idxBtfly;
			lastRealIdx = realIdx;
		}//if(idxFrame == 1) ends
		else {
			
			if(minDistance > distThres && lineIdx < 4) {
				btflyGone = true;
				//printf("Norm=%f\n", minDistance);
				//printf("Frames=%i\n", idxFrame);
			}//if(minDistance > distThres) ends
			else if(minThresCnt > 1 && lastRealIdx != realIdx)		//Occlution
				realIdx = lastRealIdx;				

		}//else ends

		int x0, y0, w, h, xmin, ymin;

		if(!btflyGone) {
			x0 = x0s[realIdx];
			y0 = y0s[realIdx];
			w = ws[realIdx];
			h = hs[realIdx];
			xmin = (int)(x0 - w / 2);
			ymin = (int)(y0 - h / 2);
			lastX0 = x0;
			lastY0 = y0;
			lastW = w;
			lastH = h;
			lastRealIdx = realIdx;

			double Ax = x0 + w / 2; 
			double Bx = x0 - w / 2;
			double Cx = x0 - w / 2; 
			double Dx = x0 + w / 2;

			double Ay = y0 - h / 2;
			double By = y0 - h / 2;
			double Cy = y0 + h / 2;
			double Dy = y0 + h / 2;		

			//sprintf(outLine, "%i %f %f %f %f %f %f %f %f\n", idxFrame, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy);
			//outFile << outLine;

			//Rect2d bbox(xmin, ymin, w, h);
			bboxReturn = new cv::Rect2d(xmin, ymin, w, h);
		}//if(!btflyGone) ends
		else
			bboxReturn = NULL;
		
  	}//if (inputFile.is_open()) ends

	return bboxReturn;
} 
 
int main(int argc, char **argv) {

	//Variables declaration
	int state;
	bool boosting = false;
    bool mil = false;
	bool kcf = false;
	bool tld = false;
	bool medianflow = false;
	bool goturn = false;
	bool mosse = false;
	bool csrt = false;

    // List of tracker types in OpenCV 3.4.1
    string trackerTypes[9] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT", "ALL"};
	string trackerColor[9] = {"blue", "green", "red", "cyan","magenta", "dark green", "yellow", "dark blue", "all"};

	int trackerIdx, btflyIdx, camIdx;

	trackerIdx = atoi(argv[1]);						//trackers index
	btflyIdx = atoi(argv[2]);						//butterfly index
	camIdx = atoi(argv[3]);							//camera index
	const string& modelFile   = argv[4];			//tracker model file as .prototxt
  	const string& trainedFile = argv[5];			//tracker trained model as .caffemodel

	char imgFileName[100];

	//Goturn tracker CNN
	Regressor regressor(modelFile, trainedFile, 0, false);

	cv::Ptr<cv::Tracker> tracker0;
	cv::Ptr<cv::Tracker> tracker1;
	cv::Ptr<cv::Tracker> tracker2;
	cv::Ptr<cv::Tracker> tracker3;
	cv::Ptr<cv::Tracker> tracker4;
	cv::Ptr<cv::Tracker> tracker5;				//Goturn tracker from opencv
	Tracker goturnTracker(false);				//Goturn tracker from code
	cv::Ptr<cv::Tracker> tracker6;
	cv::Ptr<cv::Tracker> tracker7;

	cv::VideoCapture *video;

	bool ok0, ok1, ok2, ok3, ok4, ok5, ok6, ok7; 

	 //Video frames matrix
    cv::Mat frame;

    // Define initial bounding boxes
    cv::Rect2d bbox(287, 23, 86, 320); 
	cv::Rect2d bbox0;
	cv::Rect2d bbox1;
	cv::Rect2d bbox2;
	cv::Rect2d bbox3;
	cv::Rect2d bbox4;
	cv::Rect2d *bbox5;
	BoundingBox *bboxGoturn;							//bounding box for goturn traker using its code
	cv::Rect2d bbox6;
	cv::Rect2d bbox7;
	cv::Rect2d bboxI;									//bounding box for intersection
	cv::Rect2d bboxU;									//bounding box for union
	cv::Rect2d *bboxGT = new cv::Rect2d();				//bounding box pointer for ground truth

	double IoU;										//Intersection over Union variable

	int idxFrame = 1;								//Frames index

	double hitMissThres = 0.3;						//hit-miss threshold for evaluation
	int truePositive = 0;
	int falseNegative[9] = {0};
	int falsePositive[9] = {0};
	double missRate[9] = {0.0};
	double falsePositiveRate[9] = {0.0};
	double IoUs[9] = {0.0};
	double mr, fpr, mota, motp, mt, ml;
	double mostlyTracked[9] = {0.0};
	double mostlyLost[9] = {0.0};	
	bool buflyShown = true;

	switch(trackerIdx) {
		case 0:
			printf("Boosting traker\n");
			boosting = true;
			break;
		case 1:
			printf("MIL traker\n");
			mil = true;
			break;
		case 2:
			printf("KCF traker\n");
			kcf = true;
			break;
		case 3:
			printf("TLD traker\n");
			tld = true;
			break;
		case 4:
			printf("MedianFlow traker\n");
			medianflow = true;
			break;
		case 5:
			printf("Goturn traker\n");
			goturn = true;
			break;
		case 6:
			printf("Moose traker\n");
			mosse = true;
			break;
		case 7:
			printf("CSRT traker\n");
			csrt = true;
			break;
		default:
			printf("All trakers: %i\n", trackerIdx);
			boosting = mil = kcf = tld = medianflow = goturn = mosse = csrt = true;
			break;	
	}//switch(trackerIdx) ends       

	//Variables initialization
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	state = 1;

    // Read video
	switch(camIdx) {
		case 0:
			video = new cv::VideoCapture("butterflies_videos/01-NoBlur_4Butterflies/Camera0/1-%04d.png");
			break;
		case 1:
			video = new cv::VideoCapture("butterflies_videos/01-NoBlur_4Butterflies/Camera2/2-%04d.png");
			break;
		case 2:
			video = new cv::VideoCapture("butterflies_videos/01-NoBlur_4Butterflies/Camera3/3-%04d.png");
			break;
		default:
			video = new cv::VideoCapture("butterflies_videos/01-NoBlur_4Butterflies/Camera0/1-%04d.png");
			break;
	}//switch(camIdx)  ends
     
    // Exit if video is not opened
    if(!video->isOpened()) {
        cout << "Could not read video file" << endl; 
        return 1; 
    }    

	//sprintf(outFileName, "4ButterfliesNoBlur/cameraC_alov_ann/3-%04d.ann", idxFrame + btflyIdx);
	//sprintf(fileName, "4Butterflies/cameraA_alov_ann/1-%04d.ann", idxFrame);

	//outFile.open(outFileName);
	//inputFile.open(fileName);
   
	while(state != 5) {    

		ok0 = video->read(frame); 

		if(frame.empty())
			break; 

		//sprintf(imgFileName, "4ButterfliesNoBlur/cameraC_alov/3-%04d.jpg", idxFrame);
		//imwrite(imgFileName, frame); 

		//cvtColor(frame, frame, COLOR_BGR2GRAY);
    	//GaussianBlur(frame, frame, Size(7,7), 1.5, 1.5);
    	//Canny(frame, frame, 0, 30, 3);
		//cvtColor(frame, frame, COLOR_GRAY2BGR);

		bboxGT = GetGroundBoxFromFile(btflyIdx, camIdx, idxFrame, frame.rows, frame.cols);

		if(bboxGT != NULL) {
			cv::rectangle(frame, *bboxGT, cv::Scalar(255, 255, 255), 2, 1 ); 
			cv::putText(frame, to_string(btflyIdx), cv::Point(bboxGT->x - 10, bboxGT->y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1);
			++truePositive;
			buflyShown = true;
		}//if(bboxGnd0 != NULL) ends
		else {
		
			buflyShown = false;
			break;
		
		}

		switch(state) {

			case 1:

	#if (CV_MINOR_VERSION < 3)
    {
        		tracker0 = cv::Tracker::create(trackerTypes[0]);
				tracker1 = cv::Tracker::create(trackerTypes[1]);
				tracker2 = cv::Tracker::create(trackerTypes[2]);
				tracker3 = cv::Tracker::create(trackerTypes[3]);
				tracker4 = cv::Tracker::create(trackerTypes[4]);
				tracker5 = cv::Tracker::create(trackerTypes[5]);		
				tracker6 = cv::Tracker::create(trackerTypes[6]);
				tracker7 = cv::Tracker::create(trackerTypes[7]);
	
    }
    #else
    {
        		if (boosting)
            		tracker0 = cv::TrackerBoosting::create();
        		if (mil)
            		tracker1 = cv::TrackerMIL::create();
        		if (kcf)
            		tracker2 = cv::TrackerKCF::create();
        		if (tld)
            		tracker3 = cv::TrackerTLD::create();
        		if (medianflow)
            		tracker4 = cv::TrackerMedianFlow::create();
        		if (goturn)
            		tracker5 = cv::TrackerGOTURN::create();
        		if (mosse)
            		tracker6 = cv::TrackerMOSSE::create();
        		if (csrt)
            		tracker7 = cv::TrackerCSRT::create();
    }
    #endif
				 // Uncomment the line below to select a different bounding box 
    			//bbox = selectROI(frame, false); 
				bbox = *bboxGT;
				bbox0 = bbox;
				bbox1 = bbox;
				bbox2 = bbox;
				bbox3 = bbox;
				bbox4 = bbox;
				bbox5 = &bbox;
				bbox6 = bbox;
				bbox7 = bbox;

				//destroyAllWindows();

	 			// Display bounding box.    		
				if(boosting) {
					cv::rectangle(frame, bbox0, cv::Scalar( 255, 0, 0 ), 2, 1 ); 		//blue	BOOSTING
    				tracker0->init(frame, bbox0);
				}	
				if(mil) {
					cv::rectangle(frame, bbox1, cv::Scalar( 0, 255, 0 ), 2, 1 ); 		//green	MIL
					tracker1->init(frame, bbox1);
				}
				if(kcf) {
					cv::rectangle(frame, bbox2, cv::Scalar( 0, 0, 255 ), 2, 1 ); 		//red	KCF
					tracker2->init(frame, bbox2);
				}
				if(tld) {
					cv::rectangle(frame, bbox3, cv::Scalar( 255, 255, 0 ), 2, 1 ); 		//cyan	TLD
					tracker3->init(frame, bbox3);
				}
				if(medianflow) {
					cv::rectangle(frame, bbox4, cv::Scalar( 255, 0, 255 ), 2, 1 ); 			//magenta	MEDIAN FLOW
					tracker4->init(frame, bbox4);
				}
				if(goturn) {
					cv::rectangle(frame, *bbox5, cv::Scalar( 0, 126, 0 ), 2, 1 ); 				//dark green	GOTURN
					vector<float> bboxVect{bbox5->x, bbox5->y, bbox5->x + bbox5->width, bbox5->y + bbox5->height};
					bboxGoturn = new BoundingBox(bboxVect);
					//tracker5->init(frame, bbox5);
					goturnTracker.Init(frame, *bboxGoturn, &regressor);
				}
				if(mosse) {				
					cv::rectangle(frame, bbox6, cv::Scalar( 0, 255, 255 ), 2, 1 ); 				//yellow	MOSSE
					tracker6->init(frame, bbox6);
				}

				if(csrt) {
					cv::rectangle(frame, bbox7, cv::Scalar( 126, 0, 0 ), 2, 1 );		//dark blue			CSRT
					tracker7->init(frame, bbox7);
				}

				//imshow("Tracking", frame);
				
				state = 2;

				break;
			case 2:
				 // Update the tracking result

				if(boosting)
        			ok0 = tracker0->update(frame, bbox0);

				if(mil)
					ok1 = tracker1->update(frame, bbox1);

				if(kcf)
					ok2 = tracker2->update(frame, bbox2);

				if(tld)
					ok3 = tracker3->update(frame, bbox3);

				if(medianflow)
					ok4 = tracker4->update(frame, bbox4);

				if(goturn) {
					//ok5 = tracker5->update(frame, bbox5);
					bboxGoturn = new BoundingBox();
					goturnTracker.Track(frame, &regressor, bboxGoturn);
					ok5 = true;
				}

				if(mosse)
					ok6 = tracker6->update(frame, bbox6);

				if(csrt)
					ok7 = tracker7->update(frame, bbox7);
				
				// Tracking success : Draw the tracked object

				if (ok0 && boosting) {
            		cv::rectangle(frame, bbox0, cv::Scalar( 255, 0, 0 ), 2, 1 ); 
					bboxI = bbox0 & (*bboxGT);
					bboxU = bbox0 | (*bboxGT);

					if(bboxI.area() > 0)	//Intersecting boxes
						IoU = bboxI.area() / bboxU.area();
					else	//Non-Intersecting boxes
						IoU = 0.0;

					if(IoU >= hitMissThres) {
						++mostlyTracked[0];
						IoUs[0] += IoU;
					}
					else {
						++mostlyLost[0];
						++falsePositive[0];
						++falseNegative[0];
					}//else ends

					//printf("Boosting tracker IoU:%f\n", IoU);
				}//if (ok0 && boosting) ends
				else if(boosting) {
					++falseNegative[0];
					++mostlyLost[0];
				}

				if (ok1 && mil) {
					cv::rectangle(frame, bbox1, cv::Scalar( 0, 255, 0 ), 2, 1 ); 
					bboxI = bbox1 & (*bboxGT);
					bboxU = bbox1 | (*bboxGT);

					if(bboxI.area() > 0)	//Intersecting boxes
						IoU = bboxI.area() / bboxU.area();
					else	//Non-Intersecting boxes
						IoU = 0.0;

					//printf("MIL tracker IoU:%f\n", IoU);

					if(IoU >= hitMissThres) {
						++mostlyTracked[1];
						IoUs[1] += IoU;
					}
					else {
						++mostlyLost[1];
						++falsePositive[1];
						++falseNegative[1];
					}//else ends					
				}//if (ok1 && mil) ends
				else if(mil) {
					++falseNegative[1];
					++mostlyLost[1];
				}

				if (ok2 && kcf) {
					cv::rectangle(frame, bbox2, cv::Scalar( 0, 0, 255 ), 2, 1 ); 
					bboxI = bbox2 & (*bboxGT);
					bboxU = bbox2 | (*bboxGT);

					if(bboxI.area() > 0)	//Intersecting boxes
						IoU = bboxI.area() / bboxU.area();
					else	//Non-Intersecting boxes
						IoU = 0.0;

					//printf("KCF tracker IoU:%f\n", IoU);

					if(IoU >= hitMissThres) {
						++mostlyTracked[2];
						IoUs[2] += IoU;
					}
					else {
						++mostlyLost[2];
						++falsePositive[2];
						++falseNegative[2];
					}//else ends					
				}//if (ok2 && kcf) ends
				else if(kcf) {
					++falseNegative[2];
					++mostlyLost[2];
				}

				if (ok3 && tld) {
					cv::rectangle(frame, bbox3, cv::Scalar( 255, 255, 0 ), 2, 1 ); 
					bboxI = bbox3 & (*bboxGT);
					bboxU = bbox3 | (*bboxGT);

					if(bboxI.area() > 0)	//Intersecting boxes
						IoU = bboxI.area() / bboxU.area();
					else	//Non-Intersecting boxes
						IoU = 0.0;

					//printf("TLD tracker IoU:%f\n", IoU);

					if(IoU >= hitMissThres) {
						++mostlyTracked[3];
						IoUs[3] += IoU;
					}
					else {
						++mostlyLost[3];
						++falsePositive[3];
						++falseNegative[3];
					}//else ends
				}//if (ok3 && tld)  ends
				else if(tld) {
					++falseNegative[3];
					++mostlyLost[3];
				}

				if (ok4 && medianflow) {
					cv::rectangle(frame, bbox4, cv::Scalar( 255, 0, 255 ), 2, 1 ); 
					bboxI = bbox4 & (*bboxGT);
					bboxU = bbox4 | (*bboxGT);

					if(bboxI.area() > 0)	//Intersecting boxes
						IoU = bboxI.area() / bboxU.area();
					else	//Non-Intersecting boxes
						IoU = 0.0;

					//printf("Meadian Flow tracker IoU:%f\n", IoU);

					if(IoU >= hitMissThres) {
						++mostlyTracked[4];
						IoUs[4] += IoU;
					}
					else {
						++mostlyLost[4];
						++falsePositive[4];
						++falseNegative[4];
					}//else ends
				}//if (ok3 && medianflow)  ends
				else if(medianflow) {
					++falseNegative[4];
					++mostlyLost[4];
				}			

				if (ok5 && goturn) {
					vector<float> bboxVect;
					bboxGoturn->GetVector(&bboxVect);
					bbox5 = new cv::Rect2d(bboxVect[0], bboxVect[1], bboxVect[2] - bboxVect[0], bboxVect[3] - bboxVect[1]);
					
					cv::rectangle(frame, *bbox5, cv::Scalar( 0, 126, 0 ), 2, 1 ); 
					bboxI = *bbox5 & (*bboxGT);
					bboxU = *bbox5 | (*bboxGT);

					if(bboxI.area() > 0)	//Intersecting boxes
						IoU = bboxI.area() / bboxU.area();
					else	//Non-Intersecting boxes
						IoU = 0.0;

					//printf("Goturn tracker IoU:%f\n", IoU);

					if(IoU >= hitMissThres) {
						++mostlyTracked[5];
						IoUs[5] += IoU;
					}
					else {
						++mostlyLost[5];
						++falsePositive[5];
						++falseNegative[5];
					}//else ends
				}//if (ok5 && goturn) ends
				else if(goturn) {
					++falseNegative[5];
					++mostlyLost[5];
				}

				if (ok6 && mosse) {
					cv::rectangle(frame, bbox6, cv::Scalar( 0, 255, 255 ), 2, 1 ); 
					bboxI = bbox6 & (*bboxGT);
					bboxU = bbox6 | (*bboxGT);

					if(bboxI.area() > 0)	//Intersecting boxes
						IoU = bboxI.area() / bboxU.area();
					else	//Non-Intersecting boxes
						IoU = 0.0;

					//printf("Mosse tracker IoU:%f\n", IoU);

					if(IoU >= hitMissThres) {
						++mostlyTracked[6];
						IoUs[6] += IoU;
					}
					else {
						++mostlyLost[6];
						++falsePositive[6];
						++falseNegative[6];
					}//else ends
				}//if (ok3 && mosse)  ends
				else if(mosse) {
					++falseNegative[6];
					++mostlyLost[6];
				}

				if (ok7 && csrt) {
					cv::rectangle(frame, bbox7, cv::Scalar( 126, 0, 0 ), 2, 1 );
					bboxI = bbox7 & (*bboxGT);
					bboxU = bbox7 | (*bboxGT);

					if(bboxI.area() > 0)	//Intersecting boxes
						IoU = bboxI.area() / bboxU.area();
					else	//Non-Intersecting boxes
						IoU = 0.0;

					//printf("CSRT tracker IoU:%f\n", IoU);

					if(IoU >= hitMissThres) {
						++mostlyTracked[7];
						IoUs[7] += IoU;
					}
					else {
						++mostlyLost[7];
						++falsePositive[7];
						++falseNegative[7];
					}//else ends
				}//if (ok3 && csrt)  ends
				else if(csrt) {
					++falseNegative[7];
					++mostlyLost[7];
				}

		}//switch(state)  ends		
	
        // Start timer
        double timer = (double)cv::getTickCount();      
         
        // Calculate Frames per second (FPS)
        float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
         
        // Display tracker type on frame
        cv::putText(frame, trackerTypes[trackerIdx] + " Tracker", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 1);
         
        // Display FPS on frame
        //putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);		
 
        // Display frame.
        cv::imshow("Tracking", frame);
         
        // Exit if ESC pressed.
        int k = cv::waitKey(50); //250
        
		switch(k) {

			case 82:
			case 114:
				state = 1;
				break;
			case 27:
				state = 5;
				break;

		}//switch(k) ends   

		++idxFrame;  
 
    }//while(state != 5)  ends

	if(trackerIdx != 8) {
		mr = (double)falseNegative[trackerIdx] / (double)truePositive;
		fpr = (double)falsePositive[trackerIdx] / (double)truePositive;
		mota = 1.0 - mr - fpr;
		motp = IoUs[trackerIdx] / (double)mostlyTracked[trackerIdx];
		mt = ((double)mostlyTracked[trackerIdx] / (double)truePositive) * 100;
		ml = ((double)mostlyLost[trackerIdx] / (double)truePositive) * 100;
		printf("Tracker: %s\n", trackerTypes[trackerIdx].c_str());
		printf("Color: %s\n", trackerColor[trackerIdx].c_str());
		printf("Hit-Miss threshold: %f\n", hitMissThres);
		printf("True positives: %i\n", truePositive);
		printf("False positives: %i\n", falsePositive[trackerIdx]); 
		printf("False negatives: %i\n", falseNegative[trackerIdx]);
		printf("Miss Rate (MR): %f\n", mr);
		printf("False Positive Rate (FPR): %f\n", fpr);
		printf("Multiple Object Tracking Accuracy (MOTA): %f\n", mota);
		printf("Multiple Object Tracking Precision (MOTP): %f\n", motp);
		printf("Mostly Tracked: %.2f\n", mt);
		printf("Mostly Lost: %.2f\n", ml);
	}
	else {
		for(int idxTrack = 0; idxTrack < 8; idxTrack++) {
			mr = (double)falseNegative[idxTrack] / (double)truePositive;
			fpr = (double)falsePositive[idxTrack] / (double)truePositive;
			mota = 1.0 - mr - fpr;
			motp = IoUs[idxTrack] / (double)mostlyTracked[idxTrack];
			mt = ((double)mostlyTracked[idxTrack] / (double)truePositive) * 100;
			ml = ((double)mostlyLost[idxTrack] / (double)truePositive) * 100;
			printf("Tracker: %s\n", trackerTypes[idxTrack].c_str());
			printf("Color: %s\n", trackerColor[idxTrack].c_str());
			printf("Hit-Miss threshold: %f\n", hitMissThres);
			printf("True positives: %i\n", truePositive);
			printf("False positives: %i\n", falsePositive[idxTrack]); 
			printf("False negatives: %i\n", falseNegative[idxTrack]);
			printf("Miss Rate (MR): %f\n", mr);
			printf("False Positive Rate (FPR): %f\n", fpr);
			printf("Multiple Object Tracking Accuracy (MOTA): %f\n", mota);
			printf("Multiple Object Tracking Precision (MOTP): %f\n", motp);
			printf("Mostly Tracked: %.2f\n", mt);
			printf("Mostly Lost: %.2f\n", ml);
		}//for(int idxTrack = 0; idxTrack < 8; idxTrack++) ends
	}

	//inputFile.close();
	//outFile.close();

}//int main(int argc, char **argv) ends
