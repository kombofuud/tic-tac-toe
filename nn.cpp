#include <iostream>
#include <vector>
#include <random>
#include <chrono>

//function Declarations
std::vector<std::vector<std::vector<std::vector<float>>>> makeResNet(std::vector<int>);
std::vector<std::vector<std::vector<float>>> makeResBlock(int, int);
std::vector<std::vector<float>> makeMat(int, int);
std::vector<float> activation(std::vector<float>);
void visualizeNet(std::vector<std::vector<std::vector<std::vector<float>>>>);

//random number generator initialization
std::ranlux48 ranlux48(std::chrono::system_clock::now().time_since_epoch().count());
std::normal_distribution<float> normalD(0.0, 1.0);

int main(){
    auto x = makeResNet({3,1,4});
    visualizeNet(x);
}


std::vector<std::vector<std::vector<std::vector<float>>>>  makeResNet(std::vector<int> layerSizes){ //Creates a neural network with the specified layer sizes at the end of each residual block.
    std::vector<std::vector<std::vector<std::vector<float>>>> resNet; //output vector containing a list of the residual blocks
    for(int i = 0; i < layerSizes.size()-1; i++){
        //add a residual block for each specification
        resNet.push_back(makeResBlock(layerSizes[i], layerSizes[i+1]));
    }
    return resNet;
}


std::vector<std::vector<std::vector<float>>> makeResBlock(int input, int output){
    std::vector<std::vector<std::vector<float>>> block; //A list of the 3 matrices making up the block
    block.push_back(makeMat(input, input)); //first nn layer of the block
    block.push_back(makeMat(input, output)); //second nn layer
    block.push_back(makeMat(input, output)); //scales the skip connection to match the output size
    return block;
}


std::vector<std::vector<float>> makeMat(int height, int width){ //Makes a matrix initialized using a He distribution
    const float heConst = sqrt(2.0/height);
    std::vector<std::vector<float>> columns;
    for(int i = 0; i < height; i++){
        std::vector<float> row;
        for(int j = 0; j < width; j++){
            row.push_back(heConst*normalD(ranlux48));
        }
        columns.push_back(row);
    }
    return columns;
}


void visualizeNet(std::vector<std::vector<std::vector<std::vector<float>>>> net){
    for(int i = 0; i < net.size(); i++){
        for(int j = 0; j < net[i][0].size(); j++){
            for(int k = 0; k < net[i][0][0].size(); k++){
                std::cout << net[i][0][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std:: cout << std::endl;
        for(int j = 0; j < net[i][1].size(); j++){
            for(int k = 0; k < net[i][1][0].size(); k++){
                std::cout << net[i][1][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std:: cout << std::endl;
        for(int j = 0; j < net[i][2].size(); j++){
            for(int k = 0; k < net[i][2][0].size(); k++){
                std::cout << net[i][2][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std:: cout << std::endl << std::endl;
    }
}
std::vector<float> activation(std::vector<float> input){ //Takes a list of numbers and applies an activation function to them before spitting them back out
    return input;
}