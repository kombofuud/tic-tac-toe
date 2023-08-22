#include "makeNN.cpp"
#include "runNN.cpp"

int main(){
    auto x = makeResNet({1,1});
    visualizeNet(x);
    auto y = runResNetLossless({1}, x);
    std::cerr << "Output:\n";
    visualizeNetEval(y);
}