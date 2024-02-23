/*
 * Copyright 2023-2024 Eta Compute
 * All rights reserved.
 */

#ifndef _UTILS_MODULE_H_
#define _UTILS_MODULE_H_

#include "fsl_common.h"

/*******************************************************************************
 * Definitions
 ******************************************************************************/

/*******************************************************************************
 * Exported Variables
 ******************************************************************************/

/*******************************************************************************
 * Exported Functions
 ******************************************************************************/

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

void Utils_Init(void);
void Utils_ShellInit(void);
uint32_t Utils_GetTimeInUS(void);
uint32_t Utils_CRC32(void *data_p, size_t length);
uint8_t Utils_CRC8(void *data_p, size_t length);
void Utils_GenerateRandomNumbers(void *data_p, size_t length);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* _UTILS_MODULE_H_ */
