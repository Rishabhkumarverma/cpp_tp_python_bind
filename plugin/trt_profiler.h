
#ifndef _TRT_PROFILER_H_
#define _TRT_PROFILER_H_




#include <algorithm>
#include <iostream>
#include "NvInfer.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <iterator>
#include <iomanip>


class CPyProfiler : public nvinfer1::IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;
   
    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });

        if (record == mProfile.end()) mProfile.push_back(std::make_pair(layerName, ms));
        else record->second += ms;
    }
   public:
    void printLayerTimes(const int TIMING_ITERATIONS, std::string outfile_dir)
    {
        float totalTime = 0;
        std::ofstream outfile;
        outfile.open(( outfile_dir  + "_layertime.csv"));
        //outfile.setf(std::ios::right); 
        outfile<<"NO,"<<"Layer name,"<<"Time(ms)"<<std::endl;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            outfile<<i<<":,"<<mProfile[i].first.c_str()<<","<<float(mProfile[i].second / TIMING_ITERATIONS)<<std::endl;
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
        outfile<<"Time over all layers : ,"<<  totalTime / TIMING_ITERATIONS<<std::endl;
        outfile.close();
    }
};

#endif //_FULLY_CONNECTED_H
