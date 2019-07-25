#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/tracking/tracker.hpp>


using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

vector<Rect> postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
    return boxes;
}
// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

//用随机的颜色来画这些跟踪的框
void getRandomColors(vector<Scalar> &color, int numColors)
{
    RNG rng(0);
    for(int i = 0; i<numColors;i++)
    {
        color.push_back(Scalar(rng.uniform(0,255),rng.uniform(0, 255),rng.uniform(0, 255)));
    }
    
}
Ptr<Tracker> createTrackerByName(string trackerType)
{
    Ptr<Tracker> tracker;
    if (trackerType ==  trackerTypes[0])
        tracker = TrackerBoosting::create();
    else if (trackerType == trackerTypes[1])
        tracker = TrackerMIL::create();
    else if (trackerType == trackerTypes[2])
        tracker = TrackerKCF::create();
    else if (trackerType == trackerTypes[3])
        tracker = TrackerTLD::create();
    else if (trackerType == trackerTypes[4])
        tracker = TrackerMedianFlow::create();
    else if (trackerType == trackerTypes[5])
        tracker = TrackerGOTURN::create();
    else if (trackerType == trackerTypes[6])
        tracker = TrackerMOSSE::create();
    else if (trackerType == trackerTypes[7])
        tracker = TrackerCSRT::create();
    else {
        cout << "Incorrect tracker name" << endl;
        cout << "Available trackers are: " << endl;
        for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
            std::cout << " " << *it << endl;
    }
    return tracker;
}
int main()
{
    // Load names of classes
    string classesFile = "/Users/tangxi/cpp/learnOpenCV/utils/coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    String modelConfiguration = "/Users/tangxi/cpp/learnOpenCV/utils/yolov3.cfg";
    String modelWeights = "/Users/tangxi/cpp/learnOpenCV/utils/yolov3.weights";
    
    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    string filename = "/Users/tangxi/cpp/learnOpenCV/utils/run.mp4";
    VideoCapture cap(filename);
    if(!cap.isOpened())
    {
        cerr<<"cannot open ..."<<endl;
    }
    Mat frame,blob;
    int i = 0;
    string trackerType = "CSRT";
    Ptr<MultiTracker> multiTracker = MultiTracker::create();
    vector<Scalar>colors;
    vector<Rect>bboxes;
    while(true)
    {
        cap>>frame;
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            waitKey(3000);
            break;
        }
        if(i % 5 == 0)
        {
            // Create a 4D blob from a frame.
            blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
            
            //Sets the input to the network
            net.setInput(blob);
            
            // Runs the forward pass to get output of the output layers
            vector<Mat> outs;
            net.forward(outs, getOutputsNames(net));
            bboxes = postprocess(frame,outs);
            if(bboxes.size()<1)
                continue ;

            getRandomColors(colors, bboxes.size());
            //为每个目标添加跟踪器
            for(int i = 0; i<bboxes.size();i++)
            {
                multiTracker->add(createTrackerByName(trackerType),frame,Rect2d(bboxes[i]));
            }
        }
        //跟踪
        multiTracker->update(frame);
        
        for(size_t i = 0; i<multiTracker->getObjects().size();i++)
        {
            rectangle(frame, multiTracker->getObjects()[i],  colors[i],2,1);
        }
        
        imshow("MultiTracker",frame);
        if(waitKey(1)==27)
            break;
        i++;
        
    }
    return 0;
}
