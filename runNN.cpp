#include <iostream>
#include <vector>
//#include "makeNN.cpp"

std::vector<float> runResNetLossy(std::vector<float>, std::vector<std::vector<std::vector<std::vector<float>>>> );
std::vector<std::vector<std::vector<float>>> runResNetLossless(std::vector<float>, std::vector<std::vector<std::vector<std::vector<float>>>> );
std::vector<std::vector<float>> runResBlock(std::vector<float>, std::vector<std::vector<std::vector<float>>> );
std::vector<float> biasmatMult(std::vector<float>, std::vector<std::vector<float>>);
std::vector<float> matMult(std::vector<float>, std::vector<std::vector<float>>);
std::vector<float> activation(std::vector<float>);
std::vector<float> vecAdd(std::vector<float>, std::vector<float>);
void visualizeVec(std::vector<float>);
void visualizeNetEval(std::vector<std::vector<std::vector<float>>>);



std::vector<float> runResNetLossy(std::vector<float> input, std::vector<std::vector<std::vector<std::vector<float>>>> net){//evaluate the residual network and return the output
    for(int i = 0; i < net.size(); i++){
        input = runResBlock(input, net[i])[2];
    }
    return input;
}


std::vector<std::vector<std::vector<float>>> runResNetLossless(std::vector<float> input, std::vector<std::vector<std::vector<std::vector<float>>>> net){//evaluate the residual network and return each layer in the network
    std::vector<std::vector<std::vector<float>>> output;
    for(int i = 0; i < net.size(); i++){
        output.push_back(runResBlock(input, net[i]));
    }
    return output;
}


std::vector<std::vector<float>> runResBlock(std::vector<float> input, std::vector<std::vector<std::vector<float>>> block){ //evaluates a residual block and returns the values of the three layers
    std::vector<float> midLayer;
    std::vector<float> outLayer;
    std::vector<float> rescale;
    std::vector<std::vector<float>> output;
    output.push_back(input);
    midLayer = biasmatMult(input, block[0]); //first layer
    output.push_back(midLayer);
    outLayer = biasmatMult(midLayer, block[1]);
    outLayer = activation(outLayer); //non-linear activation function
    rescale = matMult(input, block[2]); //rescales the input to the same dimensions of the output
    outLayer = vecAdd(outLayer, rescale); //residual block output
    output.push_back(outLayer);
    return output; //{input vector, midLayer vector, output vector}
}


std::vector<float> biasmatMult(std::vector<float> input, std::vector<std::vector<float>> mat){ // adds bias term to the input vector and then multiplies it by the matrix
    input.push_back(1.0); //add bias term
    std::vector<float> output;
    for(int i = 0; i < mat[0].size(); i++){ //evaluate a node in the output layer
        float sum = 0;
        for(int j = 0; j < mat.size(); j++){
            sum += mat[j][i]*input[j];
        }
        output.push_back(sum);
    }
    return output;
}


std::vector<float> matMult(std::vector<float> input, std::vector<std::vector<float>> mat){ // multiplies input vector by the matrix
    std::vector<float> output;
    for(int i = 0; i < mat[0].size(); i++){ //evaluate a node in the output layer
        float sum = 0;
        for(int j = 0; j < mat.size(); j++){
            sum += mat[j][i]*input[j];
        }
        output.push_back(sum);
    }
    return output;
}


std::vector<float> activation(std::vector<float> input){ //non-linear activation function
    for(int i = 0; i < input.size(); i++){  //relu activation function
        input[i] = (input[i] + std::abs(input[i]))/2;
    }
    return input;
}


std::vector<float> vecAdd(std::vector<float> vec1, std::vector<float> vec2){
    if(vec1.size() > vec2.size()){
        std::cerr << "vector size mismatch: " << vec1.size() << ", " << vec2.size() << " autoadjusted";
        for(int i = 0; i < vec1.size()-vec2.size(); i++)
            vec2.push_back(0);
    }
    if(vec1.size() < vec2.size()){
        std::cerr << "vector size mismatch: " << vec1.size() << ", " << vec2.size() << " autoadjusted";
        for(int i = 0; i < vec2.size()-vec1.size(); i++)
            vec1.push_back(0);
    }
    std::vector<float> output;
    for(int i = 0; i < vec1.size(); i++)
        output.push_back(vec1[i]+vec2[i]);
    return output;
}


void visualizeVec(std::vector<float> vec){
    for(int i = 0; i < vec.size(); i++){
        std::cout << vec[i] << "\n";
    }
    std::cout.flush();
}


void visualizeNetEval(std::vector<std::vector<std::vector<float>>> vec){
    std::vector<std::vector<float>> layers;
    layers.push_back(vec[0][0]);
    for(int i = 0; i < vec.size(); i++){
        layers.push_back(vec[i][1]);
        layers.push_back(vec[i][2]);
    }
    for(int i = 0; i < layers.size(); i++){
        if(i%2)
            std::cout << "\n";
        for(int j = 0; j < layers[i].size(); j++)
            std::cout << layers[i][j] << " ";
        std::cout << "\n";
    }
    std::cout.flush();
}