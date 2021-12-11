#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <fstream>

using namespace std;

//********
//#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline", "unsafe-math-optimizations")
//#pragma GCC option("arch=native", "tune=native", "no-zero-upper")

//********

#define N 784 
#define L1 1000 //midle layer
#define L2 10 
#define D 1000 //total amount of train data 
#define epochs 60000

float a = 0.02;
float TrainPixels[D][N];
float TrainLabels[D][L2];
float TestPixels[10000][N];
float TestLabels[10000][L2];
float WL1[L1][N + 1];
float WL2[L2][L1+1];
float OL1[L1];
float OL2[L2];
float EL1[L1];
float EL2[L2];


float activation_Sigmoid(float y) {
	return 1 / (1 + exp(-y));
}

float derivative_Sigmoid(float y){
	return y * (1-y);
}

void init_Weights(void) {
	int i ,y;
	for ( i = 0; i < L1; i++)
	{
		for(y =0; y< N+1 ;y++)
			WL1[i][y] = 2 * ((rand() % RAND_MAX) / (float)RAND_MAX - 0.5);

	}

	for (i = 0; i < L2; i++)
	{
		for (y = 0; y < L1+1; y++)
			WL2[i][y] = 2 * ((rand() % RAND_MAX) / (float)RAND_MAX - 0.5);

	}
	
}

void activateNN(float* Vector){
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < L1; i++) {
		OL1[i] = 0;
		for (int y = 0; y < N; y++)
			OL1[i] += WL1[i][y] * Vector[y];
		OL1[i] += WL1[i][N];
		OL1[i] = activation_Sigmoid(OL1[i]);
	}
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < L2; i++) {
		OL2[i] = 0;
		for (int y = 0; y < L1; y++)
			OL2[i] += WL2[i][y] * OL1[y];
		OL2[i] += WL2[i][L1];
		OL2[i] = activation_Sigmoid(OL2[i]);
	}
}

void calc_Error(float *target) {
	//no reason for parallelization we lose time 
	for (int i = 0; i < L2; i++) {
		EL2[i] = (OL2[i] - target[i]) * (derivative_Sigmoid(OL2[i])+a);
	}
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < L1; i++) {
		EL1[i] = 0;
		for (int i2 = 0; i2 < L2; i2++) {
			EL1[i] += EL2[i2] * WL2[i2][i] * (derivative_Sigmoid(OL1[i])+a);
		}
	}
}


void trainNN(float* input, float* target)
{


	#pragma omp parallel for schedule(static)
	for (int i = 0; i <L2 ; i++) {
		for (int j = 0; j < L1; j++) {
			WL2[i][j] -= a * EL2[i] * OL1[j];
		}
		WL2[i][L1] -= a * EL2[i];
	}
	
	

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < L1; i++) {
		for (int j = 0; j < N; j++) {
			WL1[i][j] -= a * EL1[i] * input[j];
		}
		WL1[i][N] -= a * EL1[i];
	}
	

}


void LoadTrainData(void){
	ifstream traindata("fashion-mnist_train.csv");	
	
	string label;
	string pixel[784];
	
	//skip traindata first row 
	getline(traindata, pixel[0],'\n');
	
	long int  line = 0 ;
	while(line<D){
		getline(traindata,label, ',');
		
		for(int i =0; i<783;i++)
			getline(traindata,pixel[i], ',');
		
		getline(traindata,pixel[783], '\n');
		
		
		float num = stod(label);
		
		//put data to label matrix
		
		for(int i=0; i <10; i++){
			if(num==i)
				TrainLabels[line][i] = 1;
			
			else
				TrainLabels[line][i] = 0;
		}
			
		//put data to pixel matrix
		
		for(int i=0; i <784; i++){
			TrainPixels[line][i] = stod(pixel[i]);
		}
		
		
		line++;
	}

	
	traindata.close();
}

void TrainDataAccuracy(void){
	//avrg accuracy of the hole train dataset
	
	float classifiedCorrectly = 0;
	float accuracy ;
	
	for(int y = 0;y < D;y++){
		
		activateNN(TrainPixels[y]);
		
		int classifiedLebel;
		float max = 0;
		for (int i = 0; i < L2; i++) {
			if(OL2[i]>max){
				max = OL2[i];
				classifiedLebel= i;
			}
		}
		
		int correctLebel = 0;
		for(int i =1; i<L2; i++){
			if(TrainLabels[y][i] == 1)
				correctLebel = i;
		}
		
		if(classifiedLebel == correctLebel)
			classifiedCorrectly++;
		
	}
		accuracy = classifiedCorrectly/D;
	
	printf("train data accuracy = %f \n", accuracy);
	
}

int main() {

	LoadTrainData();
	init_Weights();

	
	int rn;
	for (long int i = 0; i < epochs; i++)
	{
		
		rn = rand() % D;
		activateNN(TrainPixels[rn]);
		calc_Error(TrainLabels[rn]);
		trainNN(TrainPixels[rn], TrainLabels[rn]);
		
		if((i % 10000) == 0)
			TrainDataAccuracy();
			
	}

		
	return 0;
}
