#pragma once
#include<math.h>
#include<string.h>
#include<iostream>
#include <vector>
#include <fstream>
#include "feature_extract.h"
#define _DllExport _declspec(dllexport) //ʹ�ú궨����д��



extern "C"
{
	_DllExport float* FBank(char* file_name);
}