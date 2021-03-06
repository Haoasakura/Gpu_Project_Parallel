#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_32_atomic_functions.h"
#include <iomanip> 
#include <chrono>

using namespace std;

//saves the last move of the current player
struct lastMove {
	__int8 row; __int8 column; char player; __int8 value;
	__device__ __host__ lastMove() {}
	__device__ __host__ lastMove(__int8 _row, __int8 _column, char _player,__int8 _value) {
		row = _row;
		column = _column;
		player = _player;
		value = _value;
	}
};

class Configuration {
public:
	static const __int8 ROWS = 6;
	static const __int8 COLUMNS = 7;
	static const __int8 BOARD_SIZE = ROWS*COLUMNS;
	static const __int8 MIN_SCORE = -(ROWS*COLUMNS) / 2 + 3;
	static const __int8 MAX_SCORE = (ROWS*COLUMNS + 1) / 2 - 3;
	
	__int8 numberOfConfigurations=0;
	char* board;
	char* dev_board;
	lastMove mLastmove = lastMove(-1, -1, 'n',0);

private:
	__int8 NumberOfMoves=0;
	__int8 NumberOfStartMoves=0;

	

public:
	__device__ __host__ Configuration() {}

	__host__ Configuration(string boardConfiguration) {

		board = (char*)malloc(sizeof(char)*Configuration::BOARD_SIZE);

		for (__int8 i = 0; i <boardConfiguration.length(); i++) {
			board[i] = boardConfiguration[i];
		}

		cudaMalloc(&dev_board, sizeof(char)*Configuration::BOARD_SIZE);
		cudaMemcpy(dev_board, board, sizeof(char)*Configuration::BOARD_SIZE, cudaMemcpyHostToDevice);
		mLastmove = lastMove(-1, -1, '0',0);
		for each (char c in boardConfiguration)
		{
			if (c == 'X' || c == '0')
				NumberOfMoves++;
		}
		NumberOfStartMoves = NumberOfMoves;
	}

	friend ostream& operator<<(ostream& os, const Configuration& confg) {

		for (int i = 0; i < Configuration::ROWS; i++) {
			for (int j = 0; j < Configuration::COLUMNS; j++) {
				int idx = i*Configuration::COLUMNS + j;
				os << confg.board[idx];
			}
			os << endl;
		}
		return os;
	}

	__host__ __device__ Configuration(char* _board, lastMove _move, int numMoves, int startMoves, __int8 numConfig) {
		mLastmove.row = _move.row;
		mLastmove.column = _move.column;
		mLastmove.player = _move.player;
		mLastmove.value = -1;
		NumberOfStartMoves = startMoves;
		NumberOfMoves = numMoves + 1;
		numberOfConfigurations = numConfig;

#ifdef __CUDA_ARCH__
		dev_board = (char*)malloc(sizeof(char)*Configuration::BOARD_SIZE);
		memcpy(dev_board, _board, sizeof(char)*Configuration::BOARD_SIZE);
		if(_move.row!=-1)
			dev_board[_move.row * COLUMNS + _move.column] = _move.player;
#else
		if (_move.row != -1)
			_board[_move.row*COLUMNS + _move.column] = _move.player;
		board = new char[ROWS*COLUMNS];
		for (int i = 0; i < ROWS*COLUMNS; i++) {
			board[i] = _board[i];
		}
		cudaMalloc(&dev_board, sizeof(char)*Configuration::BOARD_SIZE);
		cudaMemcpy(dev_board, board, sizeof(char)*Configuration::BOARD_SIZE, cudaMemcpyHostToDevice);
#endif // __CUDA_ARCH__


		
	}
	//CPU Check if the last move is a winning move for the current player
	__host__ bool isWinningMove() {

		if (mLastmove.row == -1)
			return false;
		int counter = 0;

		//check the column
		for (int j = 0; j < COLUMNS; j++) {
			if (board[mLastmove.row*COLUMNS + j] == mLastmove.player) {
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
			if (board[i*COLUMNS + mLastmove.column] == mLastmove.player) {
				counter++;
				if (counter >= 4)
					return true;
			}
			else
				counter = 0;
		}
		counter = 0;
		//check right diagonal
		int ki = mLastmove.row;
		int ky = mLastmove.column;
		for (; (ki > 0 && ky > 0); ki--, ky--) {
		}
		if (!(ki >= 3 || ky >= 3)) {
			for (int k = 0; (ki + k < ROWS && ky + k < COLUMNS); k++) {
				if (board[(ki + k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if (counter >= 4) {
						return true;
					}
				}
				else {
					counter = 0;
				}
			}
		}

		//check left diagonal
		counter = 0;
		ki = mLastmove.row;
		ky = mLastmove.column;
		for (; (ki < ROWS - 1 && ky > 0); ki++, ky--) {
		}
		if (!(ki < 3 || ky > 3)) {
			for (int k = 0; (ki - k >= 0 && ky + k < COLUMNS); k++) {
				if (board[(ki - k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if (counter >= 4) {
						return true;
					}
				}
				else {
					counter = 0;
				}
			}
		}
		return false;
	}
	
	__host__ char* Configuration::getBoard() {
		char* _board = new char[ROWS*COLUMNS];

		for (int i = 0; i < ROWS*COLUMNS; i++) {
			_board[i] = board[i];
		}
		return _board;
	}
	//CPU assing a numeric value to valutate the goodness of a move checking the adjacent player paws
	__host__ int Configuration::ValutateMove(lastMove mLastmove, int pawnInARow) {
		int value = 0;
		int _value = 0;
		bool yourMove = false;
		if (mLastmove.row == -1)
			return value;

		int counter = 0;
		//check the rows
		for (int j = 0; j < COLUMNS; j++) {
			if (board[mLastmove.row*COLUMNS + j] == mLastmove.player) {
				counter++;
				if (j == mLastmove.column) {
					yourMove = true;
				}
				if (counter >= pawnInARow && yourMove) {
					if (counter == 4) {
						_value = counter + 5;
						break;
					}
					_value = counter;
				}
			}
			else {
				counter = 0;
				yourMove = false;
			}
		}
		value += _value;
		_value = 0;
		counter = 0;
		//check the columns
		for (int i = mLastmove.row; i < ROWS; i++) {
			if (board[i*COLUMNS + mLastmove.column] == mLastmove.player) {
				counter++;

				if (counter >= pawnInARow) {
					if (counter == 4) {
						_value = counter + 5;
						break;
					}
					_value = counter;
				}
			}
			else
				break;
		}

		value += _value;
		_value = 0;
		counter = 0;
		yourMove = false;
		//check right diagonal
		int ki = mLastmove.row;
		int ky = mLastmove.column;
		for (; (ki > 0 && ky > 0); ki--, ky--) {
		}
		if (!(ki >= 3 || ky >= 3)) {
			for (int k = 0; (ki + k < ROWS && ky + k < COLUMNS); k++) {
				if (board[(ki + k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if ((ki + k) == mLastmove.row && (ky + k) == mLastmove.column) {
						yourMove = true;
					}
					if (counter >= pawnInARow && yourMove) {
						if (counter == 4) {
							_value = counter + 5;
							break;
						}
						_value = counter;
					}
				}
				else {
					counter = 0;
					yourMove = false;
				}
			}
		}
		value += _value;
		_value = 0;
		counter = 0;
		yourMove = false;
		//check left diagonal
		ki = mLastmove.row;
		ky = mLastmove.column;
		for (; (ki < ROWS - 1 && ky > 0); ki++, ky--) {
		}
		if (!(ki < 3 || ky > 3)) {
			for (int k = 0; (ki - k >= 0 && ky + k < COLUMNS); k++) {
				if (board[(ki - k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if ((ki - k) == mLastmove.row && (ky + k) == mLastmove.column) {
						yourMove = true;
					}
					if (counter >= pawnInARow && yourMove) {
						if (counter == 4) {
							_value = counter + 5;
							break;
						}
						_value = counter;
					}
				}
				else {
					counter = 0;
					yourMove = false;
				}
			}
		}
		value += _value;

		return value;
	}
	//CPU assing a numeric value to valutate the goodness of a move checking the adjacent enemy paws
	__host__ int Configuration::ValutateEnemyPositions(lastMove mLastmove, int pawnInARow) {
		int value = 0;
		if (mLastmove.row == -1)
			return value;

		int counter = 0;
		bool blocked = false;
		//check the rows
		for (int j = 0; j < COLUMNS; j++) {
			if ((board[mLastmove.row*COLUMNS + j] != mLastmove.player && board[mLastmove.row*COLUMNS + j] != '-') || j == mLastmove.column) {
				counter++;
				if (j == mLastmove.column)
					blocked = true;
				if (counter >= pawnInARow) {
					if (blocked) {
						value += pawnInARow + 3;
						break;
					}
				}
			}
			else
				counter = 0;
		}

		//check the Columns
		counter = 0;
		blocked = false;
		for (int i = 0; i < ROWS; i++) {
			if ((board[i*COLUMNS + mLastmove.column] != mLastmove.player && board[i*COLUMNS + mLastmove.column] != '-') || i == mLastmove.row) {
				counter++;
				if (i == mLastmove.row)
					blocked = true;
				if (counter >= pawnInARow) {
					if (blocked) {
						value += pawnInARow + 3;
						break;
					}
				}
			}
			else
				counter = 0;
		}

		//check right diagonal
		int ki = mLastmove.row;
		int ky = mLastmove.column;
		counter = 0;
		blocked = false;
		for (; (ki > 0 && ky > 0); ki--, ky--) {
		}
		if (!(ki >= 3 || ky >= 3)) {
			for (int k = 0; (ki + k < ROWS && ky + k < COLUMNS); k++) {
				if ((board[(ki + k)*COLUMNS + (ky + k)] != mLastmove.player && board[(ki + k)*COLUMNS + (ky + k)] != '-') || ((ki + k) == mLastmove.row && (ky + k) == mLastmove.column)) {
					counter++;
					if ((ki + k) == mLastmove.row && (ky + k) == mLastmove.column)
						blocked = true;
					if (counter >= pawnInARow) {
						if (blocked) {
							value += pawnInARow + 3;
							break;
						}
					}
				}
				else
					counter = 0;
			}
		}

		//check left diagonal
		counter = 0;
		blocked = false;
		ki = mLastmove.row;
		ky = mLastmove.column;
		for (; (ki < ROWS - 1 && ky > 0); ki++, ky--) {
		}
		if (!(ki < 3 || ky > 3)) {
			for (int k = 0; (ki - k >= 0 && ky + k < COLUMNS); k++) {
				if ((board[(ki - k)*COLUMNS + (ky + k)] != mLastmove.player && board[(ki - k)*COLUMNS + (ky + k)] != '-') || ((ki - k) == mLastmove.row && (ky + k) == mLastmove.column)) {
					counter++;
					if ((ki - k) == mLastmove.row && (ky + k) == mLastmove.column)
						blocked = true;
					if (counter >= pawnInARow) {
						if (blocked) {
							value += pawnInARow + 3;
							break;
						}
					}
				}
				else
					counter = 0;
			}
		}

		return value;
	}
	//CPU Return an ordered array with all the possible moves of the next turn of the game
	__host__ lastMove* Configuration::GenerateNextMoves(char player,__int8 size) {
		__int8 idx = 0;
		
		lastMove* moves;
		moves = (lastMove*)malloc(sizeof(lastMove)*size);
		idx = 0;
		size = 0;
		for (__int8 j = 0; j < COLUMNS; j++) {
			for (__int8 i = ROWS - 1; i >= 0; i--) {
				idx = i*COLUMNS + j;
				if (board[idx] == '-') {
					moves[size++]=lastMove(i, j, player, 0);
					break;
				}
			}
		}
		return SortNextMoves(moves,size);
	}
	//CPU sort the moves in non crescent order based on the value of the moves
	__host__ lastMove* Configuration::SortNextMoves(lastMove* moves, __int8 size) {
		bool bDone = false;

		for (int i = 0; i < size; i++) {
			lastMove tmp = moves[i];
			moves[i] = moves[size/ 2 + (1 - 2 * (i % 2))*(i + 1) / 2];
			moves[size/ 2 + (1 - 2 * (i % 2))*(i + 1) / 2] = tmp;
		}


		for (int i = 0; i < size; ++i) {
			board[moves[i].row*COLUMNS + moves[i].column] = moves[i].player;
			moves[i].value = ValutateMove(moves[i], 2) + ValutateEnemyPositions(moves[i], 4);
			board[moves[i].row*COLUMNS + moves[i].column] = '-';
		}

		while (!bDone) {
			bDone = true;
			for (int i = 0; i < size - 1; ++i) {
				if (moves[i].value < moves[i + 1].value) {
					lastMove tmp = moves[i];
					moves[i] = moves[i + 1];
					moves[i + 1] = tmp;
					bDone = false;
				}
			}
		}
		return moves;
	}
	//GPU assing a numeric value to valutate the goodness of a move checking the adjacent player paws
	__device__ int Configuration::dev_ValutateMove(lastMove mLastmove, int pawnInARow) {
		int value = 0;
		int _value = 0;
		bool yourMove = false;
		if (mLastmove.row == -1)
			return value;

		int counter = 0;
		//check the rows
		for (int j = 0; j < COLUMNS; j++) {
			if (dev_board[mLastmove.row*COLUMNS + j] == mLastmove.player) {
				counter++;
				if (j == mLastmove.column) {
					yourMove = true;
				}
				if (counter >= pawnInARow && yourMove) {
					if (counter == 4) {
						_value = counter + 5;
						break;
					}
					_value = counter;
				}
			}
			else {
				counter = 0;
				yourMove = false;
			}
		}
		value += _value;
		_value = 0;
		counter = 0;
		//check the columns
		for (int i = mLastmove.row; i < ROWS; i++) {
			if (dev_board[i*COLUMNS + mLastmove.column] == mLastmove.player) {
				counter++;

				if (counter >= pawnInARow) {
					if (counter == 4) {
						_value = counter + 5;
						break;
					}
					_value = counter;
				}
			}
			else
				break;
		}

		value += _value;
		_value = 0;
		counter = 0;
		yourMove = false;
		//check right diagonal
		int ki = mLastmove.row;
		int ky = mLastmove.column;
		for (; (ki > 0 && ky > 0); ki--, ky--) {
		}
		if (!(ki >= 3 || ky >= 3)) {
			for (int k = 0; (ki + k < ROWS && ky + k < COLUMNS); k++) {
				if (dev_board[(ki + k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if ((ki + k) == mLastmove.row && (ky + k) == mLastmove.column) {
						yourMove = true;
					}
					if (counter >= pawnInARow && yourMove) {
						if (counter == 4) {
							_value = counter + 5;
							break;
						}
						_value = counter;
					}
				}
				else {
					counter = 0;
					yourMove = false;
				}
			}
		}
		value += _value;
		_value = 0;
		counter = 0;
		yourMove = false;
		//check left diagonal
		ki = mLastmove.row;
		ky = mLastmove.column;
		for (; (ki < ROWS - 1 && ky > 0); ki++, ky--) {
		}
		if (!(ki < 3 || ky > 3)) {
			for (int k = 0; (ki - k >= 0 && ky + k < COLUMNS); k++) {
				if (dev_board[(ki - k)*COLUMNS + (ky + k)] == mLastmove.player) {
					counter++;
					if ((ki - k) == mLastmove.row && (ky + k) == mLastmove.column) {
						yourMove = true;
					}
					if (counter >= pawnInARow && yourMove) {
						if (counter == 4) {
							_value = counter + 5;
							break;
						}
						_value = counter;
					}
				}
				else {
					counter = 0;
					yourMove = false;
				}
			}
		}
		value += _value;

		return value;
	}
	//GPU assing a numeric value to valutate the goodness of a move checking the adjacent enemy paws
	__device__ int Configuration::dev_ValutateEnemyPositions(lastMove mLastmove, int pawnInARow) {
		int value = 0;
		if (mLastmove.row == -1)
			return value;

		int counter = 0;
		bool blocked = false;
		//check the rows
		for (int j = 0; j < COLUMNS; j++) {
			if ((dev_board[mLastmove.row*COLUMNS + j] != mLastmove.player && dev_board[mLastmove.row*COLUMNS + j] != '-') || j == mLastmove.column) {
				counter++;
				if (j == mLastmove.column)
					blocked = true;
				if (counter >= pawnInARow) {
					if (blocked) {
						value += pawnInARow + 3;
						break;
					}
				}
			}
			else
				counter = 0;
		}

		//check the Columns
		counter = 0;
		blocked = false;
		for (int i = 0; i < ROWS; i++) {
			if ((dev_board[i*COLUMNS + mLastmove.column] != mLastmove.player && dev_board[i*COLUMNS + mLastmove.column] != '-') || i == mLastmove.row) {
				counter++;
				if (i == mLastmove.row)
					blocked = true;
				if (counter >= pawnInARow) {
					if (blocked) {
						value += pawnInARow + 3;
						break;
					}
				}
			}
			else
				counter = 0;
		}

		//check right diagonal
		int ki = mLastmove.row;
		int ky = mLastmove.column;
		counter = 0;
		blocked = false;
		for (; (ki > 0 && ky > 0); ki--, ky--) {
		}
		if (!(ki >= 3 || ky >= 3)) {
			for (int k = 0; (ki + k < ROWS && ky + k < COLUMNS); k++) {
				if ((dev_board[(ki + k)*COLUMNS + (ky + k)] != mLastmove.player && dev_board[(ki + k)*COLUMNS + (ky + k)] != '-') || ((ki + k) == mLastmove.row && (ky + k) == mLastmove.column)) {
					counter++;
					if ((ki + k) == mLastmove.row && (ky + k) == mLastmove.column)
						blocked = true;
					if (counter >= pawnInARow) {
						if (blocked) {
							value += pawnInARow + 3;
							break;
						}
					}
				}
				else
					counter = 0;
			}
		}

		//check left diagonal
		counter = 0;
		blocked = false;
		ki = mLastmove.row;
		ky = mLastmove.column;
		for (; (ki < ROWS - 1 && ky > 0); ki++, ky--) {
		}
		if (!(ki < 3 || ky > 3)) {
			for (int k = 0; (ki - k >= 0 && ky + k < COLUMNS); k++) {
				if ((dev_board[(ki - k)*COLUMNS + (ky + k)] != mLastmove.player && dev_board[(ki - k)*COLUMNS + (ky + k)] != '-') || ((ki - k) == mLastmove.row && (ky + k) == mLastmove.column)) {
					counter++;
					if ((ki - k) == mLastmove.row && (ky + k) == mLastmove.column)
						blocked = true;
					if (counter >= pawnInARow) {
						if (blocked) {
							value += pawnInARow + 3;
							break;
						}
					}
				}
				else
					counter = 0;
			}
		}

		return value;
	}
	//GPU Return an ordered array with all the possible moves of the next turn of the game
	__device__ lastMove* Configuration::dev_GenerateNextMoves(char player, __int8 size) {
		__int8 idx = 0;

		lastMove* moves;
		moves = (lastMove*)malloc(sizeof(lastMove)*size);
		idx = 0;
		size = 0;
		for (__int8 j = 0; j < COLUMNS; j++) {
			for (__int8 i = ROWS - 1; i >= 0; i--) {
				idx = i*COLUMNS + j;
				if (dev_board[idx] == '-') {
					moves[size++] = lastMove(i, j, player, 0);
					break;
				}
			}
		}
		return dev_SortNextMoves(moves, size);
	}
	//GPU sort the moves in non crescent order based on the value of the moves
	__device__ lastMove* Configuration::dev_SortNextMoves(lastMove* moves, __int8 size) {
		bool bDone = false;

		for (int i = 0; i < size; i++) {
			lastMove tmp = moves[i];
			moves[i] = moves[size / 2 + (1 - 2 * (i % 2))*(i + 1) / 2];
			moves[size / 2 + (1 - 2 * (i % 2))*(i + 1) / 2] = tmp;
		}


		for (int i = 0; i < size; ++i) {
			dev_board[moves[i].row*COLUMNS + moves[i].column] = moves[i].player;
			moves[i].value = dev_ValutateMove(moves[i], 2) + dev_ValutateEnemyPositions(moves[i], 4);
			dev_board[moves[i].row*COLUMNS + moves[i].column] = '-';
		}

		while (!bDone) {
			bDone = true;
			for (int i = 0; i < size - 1; ++i) {
				if (moves[i].value < moves[i + 1].value) {
					lastMove tmp = moves[i];
					moves[i] = moves[i + 1];
					moves[i + 1] = tmp;
					bDone = false;
				}
			}
		}
		return moves;
	}
	//return the number of moves played from the start configuration
	__device__ __host__ __int8 getNMoves() {
		return NumberOfMoves;
	}

	__device__ __host__ void setNMoves(__int8 moves) {
		NumberOfMoves = moves;
	}

	__device__ __host__ __int8 NumberStartMoves()
	{
		return NumberOfStartMoves;
	}

	__device__ __host__ ~Configuration() {
	}
};
//GPU Check if the last move is a winning move for the current player in parallel for each check
__global__ void  dev_isWinningMove(Configuration * dev_c)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	
	int counter = 0;
	int ki = 0;
	int ky = 0;
	switch (idx)
	{
	case 0:
		//check the column
		for (int j = 0; j < dev_c->COLUMNS; j++) {
			if (dev_c->dev_board[dev_c->mLastmove.row*dev_c->COLUMNS + j] == dev_c->mLastmove.player) {
				counter++;
				if (counter >= 4)
					dev_c->mLastmove.value = -3;
			}
			else
				counter = 0;
		}
		break;
	case 1:
		for (int i = 0; i < dev_c->ROWS; i++) {
			if (dev_c->dev_board[i*dev_c->COLUMNS + dev_c->mLastmove.column] == dev_c->mLastmove.player) {
				counter++;
				if (counter >= 4)
					dev_c->mLastmove.value = -3;
			}
			else
				counter = 0;
		}
		break;
	case 2:
		ki = dev_c->mLastmove.row;
		ky = dev_c->mLastmove.column;
		for (; (ki > 0 && ky > 0); ki--, ky--) {
		}
		if (!(ki >= 3 || ky >= 3)) {
			for (int k = 0; (ki + k < dev_c->ROWS && ky + k < dev_c->COLUMNS); k++) {
				if (dev_c->dev_board[(ki + k)*dev_c->COLUMNS + (ky + k)] == dev_c->mLastmove.player) {
					counter++;
					if (counter >= 4) {
						dev_c->mLastmove.value = -3;
					}
				}
				else {
					counter = 0;
				}
			}
		}
		break;
	case 3:
		ki = dev_c->mLastmove.row;
		ky = dev_c->mLastmove.column;
		for (; (ki < dev_c->ROWS - 1 && ky > 0); ki++, ky--) {
		}
		if (!(ki < 3 || ky > 3)) {
			for (int k = 0; (ki - k >= 0 && ky + k < dev_c->COLUMNS); k++) {
				if (dev_c->dev_board[(ki - k)*dev_c->COLUMNS + (ky + k)] == dev_c->mLastmove.player) {
					counter++;
					if (counter >= 4) {
						dev_c->mLastmove.value = -3;
					}
				}
				else {
					counter = 0;
				}
			}
		}
		break;
	}
}
//GPU recursive call of the PVS
__device__ int Device_Pvs(Configuration* configuration, int depth, int alpha, int beta) {

	if (configuration->mLastmove.row != -1)
	{
		dev_isWinningMove << <1, 4 >> > (configuration);
		cudaDeviceSynchronize();
	}
	
	if ((configuration->mLastmove.value==-3 && configuration->mLastmove.player == '0') || depth == 0) {
		return -(configuration->getNMoves() - configuration->NumberStartMoves());
	}

	if ((configuration->mLastmove.value == -3 && configuration->mLastmove.player == 'X') || depth == 0) {
		return (configuration->getNMoves() - configuration->NumberStartMoves());
	}

	if (configuration->getNMoves() > Configuration::ROWS*Configuration::COLUMNS - 1)
		return 0;

	__int8 ix = 0;
	__int8 size = 0;
	for (__int8 j = 0; j < Configuration::COLUMNS; j++) {
		for (__int8 i = Configuration::ROWS - 1; i >= 0; i--) {
			ix = i*Configuration::COLUMNS + j;
			if (configuration->dev_board[ix] == '-') {
				size++;
				break;
			}
		}
	}

	char nextPlayer = configuration->mLastmove.player == 'X' ? '0' : 'X';
	lastMove* moves = configuration->dev_GenerateNextMoves(nextPlayer, size);

	for (int i = 0; i<size; i++) {
		int	score;
		Configuration* c = (Configuration*)malloc(sizeof(Configuration));
		c = new Configuration(configuration->dev_board, moves[i], configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);
		if (i == 0) {

			score = -Device_Pvs(c, depth - 1, -beta, -alpha);
		}
		else {
			score = -Device_Pvs(c, depth - 1, (-alpha - 1), -alpha);
			if (alpha < score < beta)
				score = -Device_Pvs(c, depth - 1, -beta, -score);
		}

		if (score > alpha)
			alpha = score;
		if (alpha >= beta) {
			free(c);
			break;
		}
		free(c);
	}
	return alpha;
}
//CPU initial call of the PVS algorithm from the main
__global__ void Pv_Split_Init(Configuration* configuration, __int8 depth, __int8 skipColumn, __int8 alpha, __int8 beta, int* score) {
	unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < 7 && idx != skipColumn) {
		int ix = 0;
		bool validMove = false;
		Configuration* dev_c = (Configuration*)malloc(sizeof(Configuration));
		dev_c = new Configuration(configuration->dev_board, configuration->mLastmove, configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);
		for (int i = Configuration::ROWS - 1; i >= 0; i--) {
			ix = i*Configuration::COLUMNS + idx;
			if (dev_c->dev_board[ix] == '-') {
				dev_c->dev_board[ix] = dev_c->mLastmove.player;
				dev_c->mLastmove = lastMove(i, idx, dev_c->mLastmove.player, 0);
				validMove = true;
				break;
			}
		}

		if (!validMove)
			return;

		if (dev_c->mLastmove.row != -1)
		{
			dev_isWinningMove << <1, 4 >> > (dev_c);
			cudaDeviceSynchronize();
		}

		if ((dev_c->mLastmove.value == -3 && dev_c->mLastmove.player == '0') || depth == 0) {
			int t_score = -(dev_c->getNMoves() - dev_c->NumberStartMoves());
			atomicExch(score, t_score);
			return;
		}

		if ((dev_c->mLastmove.value == -3 && dev_c->mLastmove.player == 'X') || depth == 0) {
			int t_score = (dev_c->getNMoves() - dev_c->NumberStartMoves());
			atomicExch(score, t_score);
			return;
		}

		if (dev_c->getNMoves() > Configuration::ROWS*Configuration::COLUMNS - 1) {
			int t_score = 0;
			atomicExch(score, t_score);
			return;
		}
		ix = 0;
		int size = 0;
		for (__int8 j = 0; j < Configuration::COLUMNS; j++) {
			for (__int8 i = Configuration::ROWS - 1; i >= 0; i--) {
				ix = i*Configuration::COLUMNS + j;
				if (dev_c->dev_board[ix] == '-') {
					size++;
					break;
				}
			}
		}

		char nextPlayer = dev_c->mLastmove.player == 'X' ? '0' : 'X';
		lastMove* moves = dev_c->dev_GenerateNextMoves(nextPlayer, size);

		for (int i = 0; i<size; i++) {
			int	mScore;
			Configuration* c = (Configuration*)malloc(sizeof(Configuration));
			c = new Configuration(dev_c->dev_board, moves[i], dev_c->getNMoves(), dev_c->NumberStartMoves(), dev_c->numberOfConfigurations);
			if (i == 0) {
				mScore = -Device_Pvs(c, depth - 1, -beta, -alpha);
			}
			else {
				mScore = -Device_Pvs(c, depth - 1, (-alpha - 1), -alpha);
				if (alpha < mScore < beta)
					mScore = -Device_Pvs(c, depth - 1, -beta, -mScore);
			}

			if (mScore > alpha)
				alpha = mScore;
			if (alpha >= beta) {
				free(c);
				break;
			}
			free(c);
		}

		atomicMax(score, alpha);
		free(dev_c);
		return;
	}
}
//CPUcheck for a winning condition in the first move of the start configuration
int FirstSevenMoves(Configuration* configuration)
{
	char nextPlayer = configuration->mLastmove.player == 'X' ? '0' : 'X';
	lastMove *moves = configuration->GenerateNextMoves(nextPlayer,6);
	for (int i = 0; i < 6; i++)
	{
		Configuration c = Configuration(configuration->getBoard(), moves[i], configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);
		bool isWinningMove = c.isWinningMove();
		if ((isWinningMove && c.mLastmove.player == '0')) {
			return -1;
		}

		if ((isWinningMove && c.mLastmove.player == 'X')) {
			return 1;
		}
	}
	return 0;
}
//CPU recursive call of the PVS
int Pv_Split(Configuration* configuration, int depth,int alpha,int beta) {
	bool isWinningMove = configuration->isWinningMove();
	if ((isWinningMove && configuration->mLastmove.player == '0') || depth == 0) {
		return -(configuration->getNMoves() - configuration->NumberStartMoves());
	}

	if ((isWinningMove && configuration->mLastmove.player == 'X') || depth == 0) {
		return (configuration->getNMoves() - configuration->NumberStartMoves());
	}

	if (configuration->getNMoves() > Configuration::ROWS*Configuration::COLUMNS - 1)
		return 0;

	int ix = 0;
	int size = 0;
	for (__int8 j = 0; j < Configuration::COLUMNS; j++) {
		for (__int8 i = Configuration::ROWS - 1; i >= 0; i--) {
			ix = i*Configuration::COLUMNS + j;
			if (configuration->board[ix] == '-') {
				size++;
				break;
			}
		}
	}

	char nextPlayer = configuration->mLastmove.player == 'X' ? '0' : 'X';
	lastMove firstMove = configuration->GenerateNextMoves(nextPlayer,size)[0];

	int* score; int* dev_score;
	score= (int*)malloc(sizeof(int));
	cudaMalloc(&dev_score, sizeof(int));
	
	
	Configuration* c = (Configuration*)malloc(sizeof(Configuration));

	c = new Configuration(configuration->getBoard(), firstMove, configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);

	*score = -Pv_Split(c, depth - 1, -beta, -alpha);
	cudaMemcpy(dev_score, score, sizeof(int), cudaMemcpyHostToDevice);

	alpha = max(alpha, *score);
	if (alpha >= beta) {
		delete c;
		return alpha;
	}

	Configuration* dev_c;

	c = new Configuration(configuration->getBoard(), lastMove(-1, -1, nextPlayer, 0), configuration->getNMoves(), configuration->NumberStartMoves(), configuration->numberOfConfigurations);
	cudaMalloc(&dev_c, sizeof(Configuration));
	cudaMemcpy(dev_c, c, sizeof(Configuration), cudaMemcpyHostToDevice);
	Pv_Split_Init<<<1, 7>>>(dev_c,(depth-1), firstMove.column,(-alpha-1),-alpha,dev_score);
	cudaDeviceSynchronize();
	cudaMemcpy(score, dev_score, sizeof(int), cudaMemcpyDeviceToHost);
	*score = -(*score);
	
	if (alpha < *score < beta) {
		cudaMemcpy(dev_score, score, sizeof(int), cudaMemcpyHostToDevice);
		Pv_Split_Init<<<1, 7 >>>(dev_c, (depth - 1), firstMove.column, -beta, -(*score), dev_score);
		cudaDeviceSynchronize();
		cudaMemcpy(score, dev_score, sizeof(int), cudaMemcpyDeviceToHost);
		*score = -(*score);
	}
			
	alpha = max(alpha, *score);
	if (alpha >= beta) {
		delete c;
		return alpha;
	}

	delete c;
	return alpha;
}

int main() {
	string line;
	clock_t total_start;
	ifstream testFile("configurations.txt");
	ofstream writeInFileB;
	ofstream writeInFileT;
	writeInFileB.open("benchmarkerGpu.txt");
	writeInFileT.open("benchmarkerTimeGpu.txt");

	int r = 0;
	size_t* s;
	s = (size_t*)malloc(sizeof(size_t));
	cudaDeviceSetLimit(cudaLimitStackSize,16384);
	cudaDeviceGetLimit(s,cudaLimitStackSize);
	if (testFile.is_open()) {
		
		int i = 0;
		total_start = clock();
		while (getline(testFile, line)) {
			Configuration* c;	
			c = (Configuration*)malloc(sizeof(Configuration));
			c = new Configuration(line);		
			writeInFileB << *c;
			auto s= chrono::steady_clock::now();
			r = FirstSevenMoves(c);
			if (r == 0)
			{
				r = Pv_Split(c, 18, numeric_limits<int>::min(), numeric_limits<int>::max());
				if (!(r % 2 == 0))
					r = -r;
			}
			auto e = chrono::steady_clock::now();
			auto elapsed = chrono::duration_cast<std::chrono::microseconds>(e - s);
			writeInFileT << i << " "<<elapsed.count()<< " \u00B5s" <<endl;
			writeInFileB << "Configuration Number: " << i << endl;
			writeInFileB << "Time For Solve: " <<elapsed.count()<<" \u00B5s"<<endl;
			writeInFileB << "Number Of Turn Until Some Win: " << r << endl;
			writeInFileB << "________________________________" << endl;
			
			free(c);
			cudaDeviceReset();
			i++;
			if (i >250)
				break;
		}
	}

	testFile.close();
	writeInFileB.close();
	writeInFileT.close();
	cout << ((clock() - total_start) / (double)CLOCKS_PER_SEC) << endl;
	system("pause");
    return 0;
}

