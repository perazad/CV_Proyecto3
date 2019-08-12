
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
 
using namespace cv;
using namespace std;
 
// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

Rect2d GetGroundBoxFromFile(int idxBtfly, int idxFrame, int rows, int cols) {

	ifstream inputFile;
	string line;	
	int lineIdx;

	char fileName[100];

	Rect2d bboxReturn;

	sprintf(fileName, "4Butterflies/cameraA/1-%04d.txt", idxFrame);

	//cout << fileName << '\n';

	inputFile.open(fileName);

	if (inputFile.is_open()) {
		lineIdx = 0;
    	while ( getline (inputFile, line) ) {
      		//cout << line << '\n';	
			if(lineIdx == idxBtfly) {

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

				int x0 = (int)(atof(x0Str.c_str()) * cols);
				int y0 = (int)(atof(y0Str.c_str()) * rows);
				int w = (int)(atof(wStr.c_str()) * cols);
				int h = (int)(atof(hStr.c_str()) * rows);
				int xmin = (int)(x0 - w / 2);
				int ymin = (int)(y0 - h / 2);				

				Rect2d bbox(xmin, ymin, w, h);
				bboxReturn = bbox;
			}

			++lineIdx;
    	}
    	inputFile.close();
  	}

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
    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

	int trackerIdx;

	trackerIdx = atoi(argv[1]);	

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
 
    // Create a tracker
    //string trackerType = trackerTypes[2];
 
    Ptr<Tracker> tracker0;
	Ptr<Tracker> tracker1;
	Ptr<Tracker> tracker2;
	Ptr<Tracker> tracker3;
	Ptr<Tracker> tracker4;
	Ptr<Tracker> tracker5;
	Ptr<Tracker> tracker6;
	Ptr<Tracker> tracker7;

	bool ok0, ok1, ok2, ok3, ok4, ok5, ok6, ok7;
 
    #if (CV_MINOR_VERSION < 3)
    {
        tracker0 = Tracker::create(trackerTypes[0]);
		tracker1 = Tracker::create(trackerTypes[1]);
		tracker2 = Tracker::create(trackerTypes[2]);
		tracker3 = Tracker::create(trackerTypes[3]);
		tracker4 = Tracker::create(trackerTypes[4]);
		tracker5 = Tracker::create(trackerTypes[5]);		
		tracker6 = Tracker::create(trackerTypes[6]);
		tracker7 = Tracker::create(trackerTypes[7]);
	
    }
    #else
    {
        if (boosting)
            tracker0 = TrackerBoosting::create();
        if (mil)
            tracker1 = TrackerMIL::create();
        if (kcf)
            tracker2 = TrackerKCF::create();
        if (tld)
            tracker3 = TrackerTLD::create();
        if (medianflow)
            tracker4 = TrackerMedianFlow::create();
        if (goturn)
            tracker5 = TrackerGOTURN::create();
        if (mosse)
            tracker6 = TrackerMOSSE::create();
        if (csrt)
            tracker7 = TrackerCSRT::create();
    }
    #endif

	//Variables initialization
	state = 0;
    // Read video
	//VideoCapture video("butterflies1.mp4");
	VideoCapture video("4Butterflies/cameraA/1-%04d.png");
     
    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl; 
        return 1; 
    } 
 
    // Read first frame 
    Mat frame;     
 
    // Define initial bounding box 
    Rect2d bbox(287, 23, 86, 320); 
	Rect2d bbox0;
	Rect2d bbox1;
	Rect2d bbox2;
	Rect2d bbox3;
	Rect2d bbox4;
	Rect2d bbox5;
	Rect2d bbox6;
	Rect2d bbox7;
	Rect2d bboxGnd;

	int idxFrame = 1;
   
	while(state != 5) {    

		ok0 = video.read(frame); 

		if(frame.empty())
			break; 

		//cvtColor(frame, frame, COLOR_BGR2GRAY);
    	//GaussianBlur(frame, frame, Size(7,7), 1.5, 1.5);
    	//Canny(frame, frame, 0, 30, 3);
		//cvtColor(frame, frame, COLOR_GRAY2BGR);

		switch(state) {

			case 1:
				 // Uncomment the line below to select a different bounding box 
    			bbox = selectROI(frame, false); 
				bbox0 = bbox;
				bbox1 = bbox;
				bbox2 = bbox;
				bbox3 = bbox;
				bbox4 = bbox;
				bbox5 = bbox;
				bbox6 = bbox;
				bbox7 = bbox;

				destroyAllWindows();

	 			// Display bounding box.    		
				if(boosting) {
					rectangle(frame, bbox0, Scalar( 255, 0, 0 ), 2, 1 ); 		//blue	BOOSTING
    				tracker0->init(frame, bbox0);
				}	
				if(mil) {
					rectangle(frame, bbox1, Scalar( 0, 255, 0 ), 2, 1 ); 		//green	MIL
					tracker1->init(frame, bbox1);
				}
				if(kcf) {
					rectangle(frame, bbox2, Scalar( 0, 0, 255 ), 2, 1 ); 		//red	KCF
					tracker2->init(frame, bbox2);
				}
				if(tld) {
					rectangle(frame, bbox3, Scalar( 255, 255, 0 ), 2, 1 ); 		//cyan	TLD
					tracker3->init(frame, bbox3);
				}
				if(medianflow) {
					rectangle(frame, bbox4, Scalar( 255, 0, 255 ), 2, 1 ); 			//magenta	MEDIAN FLOW
					tracker4->init(frame, bbox4);
				}
				if(goturn) {
					rectangle(frame, bbox5, Scalar( 0, 126, 0 ), 2, 1 ); 				//dark green	GOTURN
					tracker5->init(frame, bbox5);
				}
				if(mosse) {				
					rectangle(frame, bbox6, Scalar( 0, 255, 255 ), 2, 1 ); 				//yellow	MOSSE
					tracker6->init(frame, bbox6);
				}

				if(csrt) {
					rectangle(frame, bbox7, Scalar( 126, 0, 0 ), 2, 1 );		//dark blue			CSRT
					tracker7->init(frame, bbox7);
				}

				imshow("Tracking", frame);
				
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

				if(goturn)
					ok5 = tracker5->update(frame, bbox5);

				if(mosse)
					ok6 = tracker6->update(frame, bbox6);

				if(csrt)
					ok7 = tracker7->update(frame, bbox7);
				
				// Tracking success : Draw the tracked object

				if (ok0 && boosting) 
            		rectangle(frame, bbox0, Scalar( 255, 0, 0 ), 2, 1 ); 

				if (ok1 && mil) 
					rectangle(frame, bbox1, Scalar( 0, 255, 0 ), 2, 1 ); 

				if (ok2 && kcf) 
					rectangle(frame, bbox2, Scalar( 0, 0, 255 ), 2, 1 ); 

				if (ok3 && tld) 
					rectangle(frame, bbox3, Scalar( 255, 255, 0 ), 2, 1 ); 

				if (ok4 && medianflow) 
					rectangle(frame, bbox4, Scalar( 255, 0, 255 ), 2, 1 ); 

				if (ok5 && goturn) 
					rectangle(frame, bbox5, Scalar( 0, 126, 0 ), 2, 1 ); 

				if (ok6 && mosse) 
					rectangle(frame, bbox6, Scalar( 0, 255, 255 ), 2, 1 ); 

				if (ok7 && csrt) 
					rectangle(frame, bbox7, Scalar( 126, 0, 0 ), 2, 1 );

		}//switch(state)  ends		
	
        // Start timer
        double timer = (double)getTickCount();      
         
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        // Display tracker type on frame
        //putText(frame, trackerTypes[7] + " Tracker", Point(20,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255,0,0),2);
         
        // Display FPS on frame
        //putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

		if(state == 2) {
			bboxGnd = GetGroundBoxFromFile(0, idxFrame, frame.rows, frame.cols);
			rectangle(frame, bboxGnd, Scalar(255, 255, 255), 2, 1 ); 
		}
 
        // Display frame.
        imshow("Tracking", frame);
         
        // Exit if ESC pressed.
        int k = waitKey(250);
        
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

}//int main(int argc, char **argv) ends
