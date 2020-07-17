#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <unistd.h>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "img_process.h"
#include "post_process.h"


using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;
using namespace post_process;
int CURRENT_DIM = 416;
int BATCH_SIZE = 1;

vector<vector<int> > output_shape = { {1, 21, 13, 13}, {1, 21, 26, 26} };
vector<vector<int> > g_masks = { {3, 4, 5}, {0, 1, 2} };
vector<vector<int> > g_anchors = { {10, 14}, {23, 27}, {37, 58}, {81, 82}, {135, 169}, {344, 319} };
float obj_threshold = 0.5;
float nms_threshold = 0.4;
int CATEGORY = 2;
vector<string> labels = { "people","box"};


vector<vector<DetectionRes> > postProcess(cv::Mat& image, float * output) {
    vector<DetectionRes> detections2;
    int total_size = 0;
    /*
     * 一共有几个输出，yolov3tiny是两个。
     * 然后遍历每个输出的结果，并保存。
     */
    for (int i = 0; i < output_shape.size(); i++) {
        auto shape = output_shape[i];
        int size = 1;
        for (int j = 0; j < shape.size(); j++) {
            size *= shape[j];
        }
        total_size += size;
    }

    /*
     * 注意到输出内容是被拉成了一维向量，存储在output里面的，需要按照一定的顺序将其取出来
     */
    int offset = 0;
    float * transposed_output = new float[total_size];
    float * transposed_output_t = transposed_output;
    for (int i = 0; i < output_shape.size(); i++) {
        auto shape = output_shape[i];  // nchw
        int chw = shape[1] * shape[2] * shape[3];
        int hw = shape[2] * shape[3];
        for (int n = 0; n < shape[0]; n++) {
            int offset_n = offset + n * chw;
            for (int h = 0; h < shape[2]; h++) {
                for (int w = 0; w < shape[3]; w++) {
                    int h_w = h * shape[3] + w;
                    for (int c = 0; c < shape[1]; c++) {
                        int offset_c = offset_n + hw * c + h_w;
                        *transposed_output_t++ = output[offset_c];
                    }
                }
            }
        }
        offset += shape[0] * chw;
    }

    vector<vector<int> > shapes;
    for (int i = 0; i < output_shape.size(); i++) {
        auto shape = output_shape[i];
        vector<int> tmp = { shape[2], shape[3], 3, 4+1+CATEGORY };
        shapes.push_back(tmp);
    }



    offset = 0;
    vector<float> label_score;
    vector<vector<DetectionRes> > all_detections(CATEGORY);
    for (int i = 0; i < output_shape.size(); i++) { // batch size
        auto masks = g_masks[i];
        vector<vector<int> > anchors;
        for (auto mask : masks)
            anchors.push_back(g_anchors[mask]);

        auto shape = shapes[i];


        for (int h = 0; h < shape[0]; h++) {
            int offset_h = offset + h * shape[1] * shape[2] * shape[3];
            for (int w = 0; w < shape[1]; w++) {
                int offset_w = offset_h + w * shape[2] * shape[3];
                for (int c = 0; c < shape[2]; c++) {
                    int offset_c = offset_w + c * shape[3];
                    //std::cout<<"offset_c"<<offset_c<<std::endl;
                    // 7位7位往下走
                    float * ptr = transposed_output + offset_c;

                    ptr[4] = post_process::sigmoid(ptr[4]); // conf
                    ptr[5] = post_process::sigmoid(ptr[5]); // label1
                    ptr[6] = post_process::sigmoid(ptr[6]); // label2
                    //float score = ptr[4] * ptr[5];
                    if (ptr[4] < obj_threshold)
                        continue;
                    // 这里简单来说就是在做yololayer的步骤
                    // 将基于anchors预测到的值，还原到在输入图上（416*416），基于右上角的坐标
                    ptr[0] = (post_process::sigmoid(ptr[0])+w)*CURRENT_DIM/shape[0]; // center x
                    ptr[1] = (post_process::sigmoid(ptr[1])+h)*CURRENT_DIM/shape[1]; // center y
                    ptr[2] = post_process::exponential(ptr[2]) * anchors[c][0]; // cal w
                    ptr[3] = post_process::exponential(ptr[3]) * anchors[c][1]; // cal h


                    DetectionRes det;
                    det.prob = ptr[4];
                    det.label1 = ptr[5];
                    det.label2 = ptr[6];
                    det.x1 = ptr[0]-ptr[2]/2;
                    det.y1 = ptr[1]-ptr[3]/2;
                    det.x2 = ptr[0]+ptr[2]/2;
                    det.y2 = ptr[1]+ptr[3]/2;
                    det.score = ptr[4]*std::max(ptr[5],ptr[6]);
                    label_score.push_back(ptr[5]);
                    label_score.push_back(ptr[6]);
                    det.max_label = std::max(ptr[5],ptr[6]);
                    det.max_id = post_process::arg_max(label_score);
                    label_score.clear();

                    detections2.push_back(det);

                    all_detections[det.max_id].push_back(det);
                }
            }
        }
        offset += shape[0] * shape[1] * shape[2] * shape[3];
    }


    for(int i=0;i<all_detections.size();i++){
        // 每个类别排序，然后做一遍NMS
        sort(all_detections[i].begin(), all_detections[i].end(), [=](const DetectionRes & left, const DetectionRes & right) {
            return left.score > right.score;
        });
        DoNms(all_detections[i],nms_threshold);
    }
    // 返回
    return all_detections;
}

void load_Onnx_Serialize(const char* &onnx_filename,string & engine_path){
    // 1 加载onnx模型
    IBuilder* builder = createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    //const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    //INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    //const char* onnx_filename="/home/webank/07_clion/TensorRT_local/weights/yolov3-mytiny_98_0.96_warehouse.onnx";
    //parser->parseFromFile(onnx_filename, 2);
    parser->parseFromFile(onnx_filename, static_cast<int>(Logger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    std::cout << "successfully load the onnx model" << std::endl;

    // 2、build the engine
    unsigned int maxBatchSize=1;
    builder->setMaxBatchSize(maxBatchSize);
    // 所以其实是这个设置不好使了，需要用那个config的嘛？
    // builder->setFp16Mode(true);

    BuilderFlag bflag = BuilderFlag::kFP16;

    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 26);
    config->setFlag(bflag);
    std::cout<<"mode flag:   "<<config->getFlag(bflag)<<std::endl;

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // 3、serialize Model
    IHostMemory *gieModelStream = engine->serialize();
    std::string serialize_str;
    std::ofstream serialize_output_stream;
    serialize_str.resize(gieModelStream->size());
    memcpy((void*)serialize_str.data(),gieModelStream->data(),gieModelStream->size());
    serialize_output_stream.open(engine_path);
    serialize_output_stream<<serialize_str;
    serialize_output_stream.close();

    parser->destroy();
    engine->destroy();
    network->destroy();
    builder->destroy();

}

void load_engine(std::string cached_path,ICudaEngine* &re_engine){
    IRuntime* runtime = createInferRuntime(gLogger);
    std::ifstream fin(cached_path);
    std::string cached_engine = "";
    while (fin.peek() != EOF){
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    re_engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}





int main() {

    ICudaEngine* engine;

    // 4、deserialize model
    const char* onnx_filename="/home/webank/07_clion/TensorRT_local/weights/yolov3-mytiny_98_0.96_warehouse.onnx";
    std::string cached_path = "/home/webank/07_clion/TensorRT/serialize_engine_output.trt";
    std::fstream existEngine;
    existEngine.open(cached_path, ios::in);
    if(existEngine){
        load_engine(cached_path,engine);
        std::cout<<"load model from engine"<<std::endl;
    }else{
        load_Onnx_Serialize(onnx_filename,cached_path);
        load_engine(cached_path,engine);
        std::cout<<"load model from onnx and convert to engine"<<std::endl;

    }

    assert(engine!=nullptr);
    IExecutionContext* context = engine->createExecutionContext();


    //get buffers
    // yolov3 4 ; yolov3-tiny 3
    int target_bind_nb = 3;
    assert(engine->getNbBindings() == target_bind_nb);
    void* buffers[target_bind_nb];
    std::vector<int64_t> bufferSize;
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i)
    {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        CHECK(cudaMalloc(&buffers[i], totalSize));
    }


    //get stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // prepare output

    vector<cv::Mat> inputImgs;
    int outSize1 = bufferSize[1] / sizeof(float);
    int outSize2 = bufferSize[2] / sizeof(float);
    float* out1 = new float[outSize1];
    float* out2 = new float[outSize2];

    //start to do inference
    int index = 1,batchCount = 0;

    // 读取图片
    std::string img_path  ="/home/webank/07_clion/TensorRT_local/data/warehouse1.jpg";
    cv::Mat img = cv::imread(img_path);
    cv::Mat org_img = cv::imread(img_path);
    cv::Mat float_img;
    // 图片预处理
    mypreprocess(img,float_img,CURRENT_DIM);

    // 图片数据转为inputs
    int c=3,w=416,h=416;
    vector<Mat> input_channels(c);
    cv::split(float_img, input_channels);

    vector<float> result(h * w * c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }
    vector<float> curInput = result;

    CHECK(cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream));


    auto t_start = std::chrono::high_resolution_clock::now();
    context->execute(BATCH_SIZE, buffers);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Inference take: " << total << " ms." << endl;

    CHECK(cudaMemcpyAsync(out1, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(out2, buffers[2], bufferSize[2], cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    float* out = merge(out1, out2, outSize1, outSize2);
    auto all_detections = postProcess(org_img, out);

    std::cout<<org_img.rows<<" "<<org_img.cols<<std::endl;
    int height = org_img.rows;
    int width = org_img.cols;
    //print boxes
    for(int i=0;i<all_detections.size();i++){

        for(int j=0;j<all_detections[i].size();j++){
            all_detections[i][j]=rescale(all_detections[i][j],height,width,CURRENT_DIM);
            std::cout<<"i: "<<i<<", j: "<<j<<std::endl;
            std::cout << all_detections[i][j].x1 << ", " << all_detections[i][j].y1 << ", " << all_detections[i][j].x2<< ", "<< all_detections[i][j].y2<<", "<<all_detections[i][j].prob<<",  "<<all_detections[i][j].max_label<<", "<<all_detections[i][j].max_id<<std::endl;
            cv::rectangle(org_img, cv::Point(all_detections[i][j].x1, all_detections[i][j].y1), cv::Point(all_detections[i][j].x2, all_detections[i][j].y2), cv::Scalar(0, 0, 255), 1, 1, 0);

        }
    }

    cv::imwrite("out-det_0.jpg", org_img);


    std::cout << "Hello, World!" << std::endl;
    return 0;
}
