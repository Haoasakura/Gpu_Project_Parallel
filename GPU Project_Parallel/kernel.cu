#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <cstdint>
#include <cassert>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_32_atomic_functions.h"

using namespace std;

__device__ int gAlpha=-100;
__device__ int gBeta=100;
__device__ int gScore=0;

struct lastMove {
	int row; int column; char player; int value;
	lastMove(int _row, int _column, char _player, int _value) {
		row = _row;
		column = _column;
		player = _player;
		value = _value;
	}
};

class Configuration {
public:
	static const int ROWS = 6;  // width of the board
	static const int COLUMNS = 7; // height of the board
	static const int BOARD_SIZE = ROWS*COLUMNS;
	static const int MIN_SCORE = -(ROWS*COLUMNS) / 2 + 3;
	static const int MAX_SCORE = (ROWS*COLUMNS + 1) / 2 - 3;
	
	int numberOfConfigurations=0;
	char* board;
	char* dev_board;
	lastMove mLastmove = lastMove(-1, -1, 'n', 0);
private:
	int NumberOfMoves;
	
	int NumberOfStartMoves;

	

public:
	__device__ __host__ Configuration() {}
	Configuration(string boardConfiguration) {

		board = (char*)malloc(sizeof(char)*Configuration::BOARD_SIZE);

		for (int i = 0; i <boardConfiguration.length(); i++) {
			board[i] = boardConfiguration[i];
		}

		cudaMalloc(&dev_board, sizeof(char)*Configuration::BOARD_SIZE);
		cudaMemcpy(dev_board, board, sizeof(char)*Configuration::BOARD_SIZE, cudaMemcpyHostToDevice);

		mLastmove = lastMove(-1, -1, '0', 0);
		for each (char c in boardConfiguration)
		{
			if (c == 'X' || c == '0')
				NumberOfMoves++;
		}
		NumberOfStartMoves = NumberOfMoves;
	}



	__device__ __host__ void PrintBoard() {
		for (int i = 0; i < Configuration::ROWS; i++) {
			for (int j = 0; j < Configuration::COLUMNS; j++) {
				printf("%c", board[i*Configuration::COLUMNS+j]);
			}
			printf("\n");
		}
	}


	__device__ __host__ ~Configuration() {
	}
};

__global__ void BoardPrint(Configuration *c)
{

	Configuration* tmp = new Configuration[1];
	memcpy(&tmp[0],c,sizeof(Configuration));
	for (int i = 0; i < Configuration::ROWS; i++) {
		for (int j = 0; j < Configuration::COLUMNS; j++) {
			int idx = i*Configuration::COLUMNS + j;
			printf("%c", tmp[0].dev_board[idx]);
		}
	}
}

int main()
{
	string line;
	double duration;
	ifstream testFile("configurations.txt");
	ofstream writeInFile;
	writeInFile.open("benchmarker.txt");
	if (testFile.is_open()) {
		int i = 0;
		while (getline(testFile, line)) {
			Configuration *dev_c, *c;
			c = (Configuration*)malloc(sizeof(Configuration));
			c = new Configuration(line);
			/*c->board = (char*)malloc(sizeof(char)*Configuration::BOARD_SIZE);

			for (int i = 0; i <line.length(); i++) {
				c->board[i] = line[i];
			}

			cudaMalloc(&c->dev_board, sizeof(char)*Configuration::BOARD_SIZE);
			cudaMemcpy(c->dev_board, c->board, sizeof(char)*Configuration::BOARD_SIZE, cudaMemcpyHostToDevice);*/

			cudaMalloc(&dev_c, sizeof(Configuration));
			cudaMemcpy(dev_c, c, sizeof(Configuration), cudaMemcpyHostToDevice);

			BoardPrint << <1, 1 >> > (dev_c);

			cudaDeviceReset();
			i++;
			if (i >0)
				break;
		}
	}

	
	system("pause");
    return 0;
}