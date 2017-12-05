#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_32_atomic_functions.h"

using namespace std;

__device__ __managed__ int gAlpha=-100;
__device__ __managed__ int gBeta=100;
__device__ __managed__ int gScore=0;
__device__ __managed__ unsigned int nodeCount = 0;

struct lastMove {
	int row; int column; char player; int value;
	__device__ __host__ lastMove() {}
	__device__ __host__ lastMove(int _row, int _column, char _player, int _value) {
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
	int NumberOfMoves=0;
	int NumberOfStartMoves=0;

	

public:
	__device__ __host__ Configuration() {}
	__host__ Configuration(string boardConfiguration) {

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

	__device__  Configuration(char* _dev_board, lastMove _move, int numMoves, int startMoves, int numConfig) {
		mLastmove.row = _move.row;
		mLastmove.column = _move.column;
		mLastmove.player = _move.player;
		NumberOfStartMoves = startMoves;
		NumberOfMoves = numMoves + 1;
		numberOfConfigurations = numConfig;

		dev_board = (char*)malloc(sizeof(char)*Configuration::BOARD_SIZE);
		memcpy(dev_board,_dev_board,sizeof(char)*Configuration::BOARD_SIZE);
		dev_board[_move.row * COLUMNS + _move.column] = _move.player;
	}

	__device__ bool isFull() {

		int idx = 0;
		int counter = 0;
		for (int j = 0; j < COLUMNS; j++) {
			for (int i = ROWS - 1; i >= 0; i--) {
				idx = i*COLUMNS + j;
				if (dev_board[idx] == '-') {
					counter++;
					break;
				}
			}
		}

		numberOfConfigurations = counter;
		if (numberOfConfigurations > 0)
			return false;
		else
			return true;
	}

	__device__ bool isWinningMove() {

		if (mLastmove.row == -1)
			return false;
		int counter = 0;
		//check the column
		for (int j = 0; j < COLUMNS; j++) {
			if (dev_board[mLastmove.row*COLUMNS + j] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		//check the row
		for (int i = 0; i < ROWS; i++) {
			if (dev_board[i*COLUMNS + mLastmove.column] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		//check right diagonal
		for (int k = 0; (k + mLastmove.row < ROWS && k + mLastmove.column < COLUMNS); k++) {
			if (dev_board[(mLastmove.row + k)*COLUMNS + (mLastmove.column + k)] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		for (int k = 0; (mLastmove.row - k >= 0 && mLastmove.column - k >= 0); k++) {
			if (dev_board[(mLastmove.row - k)*COLUMNS + (mLastmove.column - k)] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		//check left diagonal
		counter = 0;
		for (int k = 0; (k + mLastmove.row < ROWS && mLastmove.column - k >= 0); k++) {
			if (dev_board[(mLastmove.row + k)*COLUMNS + (mLastmove.column - k)] == mLastmove.player) {
				counter++;

				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		for (int k = 0; (mLastmove.row - k >= 0 && k + mLastmove.column < COLUMNS); k++) {
			if (dev_board[(mLastmove.row - k)*COLUMNS + (mLastmove.column + k)] == mLastmove.player) {
				counter++;

				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		return false;
	}

	__device__ __host__ void PrintBoard() {
		for (int i = 0; i < Configuration::ROWS; i++) {
			for (int j = 0; j < Configuration::COLUMNS; j++) {
				int idx = i*Configuration::COLUMNS + j;
				printf("%c", board[idx]);
			}
			printf("\n");
		}
	}

	__device__ int getNMoves() {
		return NumberOfMoves;
	}

	__device__ void setNMoves(int moves) {
		NumberOfMoves = moves;
	}

	__device__ int NumberStartMoves()
	{
		return NumberOfStartMoves;
	}

	__device__ __host__ ~Configuration() {
	}
};

__global__ void BoardPrint(Configuration *c)
{
	for (int i = 0; i < Configuration::ROWS; i++) {
		for (int j = 0; j < Configuration::COLUMNS; j++) {
			int idx = i*Configuration::COLUMNS + j;
			printf("%c", c->dev_board[idx]);
		}
		printf("\n");
	}
}

__global__ void MiniMax(Configuration* configuration, int depth) {
	int thread = threadIdx.x;
	atomicAdd(&nodeCount,1);
	bool freeSpace = false;
	Configuration* c;
	c = (Configuration*)malloc(sizeof(Configuration));
	char nextPlayer = configuration->mLastmove.player == 'X' ? '0' : 'X';
	for (int i = Configuration::ROWS - 1; i >= 0; i--) {
		int idx = i*Configuration::COLUMNS + thread;
		if (configuration->dev_board[idx] == '-') {
			c = new Configuration(configuration->dev_board, lastMove(i, thread, nextPlayer, 0), configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);
			freeSpace = true;
			break;
		}
	}

	if (freeSpace && thread < 7) {

		bool isWinningMove = c->isWinningMove();
		if (isWinningMove && c->mLastmove.player == '0') {
			int losingScore = -(c->getNMoves() - c->NumberStartMoves());
			if (losingScore < gBeta)
				atomicMin(&gBeta, losingScore);
		}
		if (isWinningMove && c->mLastmove.player == 'X')
		{
			int winningScore = (c->getNMoves() - c->NumberStartMoves());
			if (winningScore > gAlpha)
				atomicMax(&gAlpha, winningScore);
		}

		if (c->getNMoves() > Configuration::ROWS*Configuration::COLUMNS - 1)
		{
			int drawScore = 0;
			if (drawScore > gAlpha )
				atomicMax(&gAlpha, drawScore);
		}

		//int min = -(Configuration::ROWS*Configuration::COLUMNS - 2 - configuration.getNMoves()) / 2;
		if (alpha <= gAlpha)
			alpha = gAlpha;
		else
			//__SM_32_ATOMIC_FUNCTIONS_H__::max(gAlpha,alpha);
			atomicMax(&gAlpha, alpha);

		//int max = (Configuration::ROWS*Configuration::COLUMNS - 1 - configuration.getNMoves()) / 2;
		if (beta > gBeta)
			beta = gBeta;
		else
			//__SM_32_ATOMIC_FUNCTIONS_H__::min(gBeta, beta);
			atomicMin(&gBeta, beta);

		if (gAlpha >= gBeta)
			return;		

		if (nextPlayer == 'X') {
			if (depth > 0 && !configuration->isFull()) {
				int score = alpha;
				MiniMax << <1, 7 >> > (c, depth - 1, alpha, beta);
				cudaDeviceSynchronize();
				/*if (score < alpha) {
					score = alpha;
				}*/

				if (score < gAlpha)
					atomicMin(&gAlpha, score);
				if (gBeta <= gAlpha) {
					gScore = score;

				}

				printf("%d  ", gAlpha);
			}
		}

		if (nextPlayer == '0') {
			if (depth > 0 && !configuration->isFull()) {
				int score = beta;
				MiniMax << <1, 7 >> > (c, depth - 1, alpha, beta);
				cudaDeviceSynchronize();
				/*if (score > beta) {
					score = beta;
				}*/

				if (score >= gBeta)
					atomicMax(&gBeta, score);
				if (gBeta <= gAlpha) {
					gScore = score;
				}

				printf("%d  ", gBeta);
			}
		}

	}
}


int main() {
	string line;
	//double duration;
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

			//BoardPrint << <1, 1 >> > (dev_c);
			MiniMax << <1, 7 >> > (dev_c,10,-100,100);
			cudaDeviceSynchronize();

			//cudaMemcpy(&alpha,&gAlpha,sizeof(int),cudaMemcpyDeviceToHost);
			//cudaMemcpy(&beta, &gBeta, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaMemcpy(&score, &gScore, sizeof(int), cudaMemcpyDeviceToHost);
			printf("Configuration N  %d \n", i);
			printf("galpha %d \n", gAlpha);
			printf("gbeta %d \n", gBeta);
			printf("gscore %d \n", gScore);
			printf("nodes %u \n", nodeCount);
			printf("-----------------------------------\n");

			free(c);
			cudaFree(dev_c);
			gAlpha = -100;
			gBeta = 100;
			gScore = 0;
			nodeCount = 0;
			cudaDeviceReset();		
			i++;
			if (i >1)
				break;
		}
	}

	
	system("pause");
    return 0;
}