#include <stdlib.h>
#include <iostream>
#include "pcm_utils.h"
#include "DTLN_NS_ncnn.h"

#include "ncnn/benchmark.h"

int main(int argc, char **argv)
{
    if(argc!=4)
    {
        std::cout<<"usage: ./dtln <modeldir> <in.pcm> <out.pcm>"<<std::endl;
        return 0;
    }

    ns::DTLN_NS_ncnn alg;
    std::string modelDir = argv[1];
    std::vector<std::string> modelPaths ={
            modelDir+"/dtln_ns_p1.ncnn.param",
            modelDir+"/dtln_ns_p1.ncnn.bin",
            modelDir+"/dtln_ns_p2.ncnn.param",
            modelDir+"/dtln_ns_p2.ncnn.bin",
    };
    alg.load_model(modelPaths);

    const float fs = 32000;
    const int chunk_size = alg.trunk_size();
    const int chunk_size_in_bytes = alg.trunk_size_in_bytes();

    const char  * filepath = argv[2];
    PCMReader  pcmReader(filepath);

    long  inputSamples = pcmReader.sample_size_in_16bit();
    std::cout<<"audio secs: "<<inputSamples / fs <<std::endl;

    long  totalTrunks = inputSamples / chunk_size;
    float chunkSizeMS = 1000* chunk_size /fs;
    std::cout<<"chunk size in ms: "<<  chunkSizeMS <<std::endl;
    std::cout<<"audio total chunks: "<<totalTrunks <<std::endl;


    const char  * dstpath = argv[3];
    PCMWriter  pcmWriter(dstpath);

    char * inbuf = new char [chunk_size_in_bytes];
    char * outbuf = new char [chunk_size_in_bytes];

    double  costAvg = 0.0f;
    for(int i = 0; i < totalTrunks; i++)
    {
        pcmReader.read_bytes(inbuf, chunk_size_in_bytes);
        double start= ncnn::get_current_time();
        alg.process_1_channel_trunk_by_trunk(inbuf, outbuf, chunk_size_in_bytes);
        double cost = ncnn::get_current_time() - start;
        costAvg += cost;
        pcmWriter.write_bytes(outbuf, chunk_size_in_bytes);

    }
    costAvg /= totalTrunks;
    std::cout<<"one trunk avg cost: "<<costAvg<<" ms"<<std::endl;
    std::cout<<"RTF (trunk time/ run time): "<< chunkSizeMS / costAvg<<std::endl;

    delete [] inbuf;
    delete [] outbuf;

    return 0;
}