/*
 * Copyright (c) 2015, Freescale Semiconductor, Inc.
 * Copyright 2016-2023 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tensorflow/lite/micro/kernels/softmax.h"

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include <stdint.h>
#include "tensorflow/lite/micro/fsl_powerquad.h"

namespace tflite {

    namespace {
    enum {
        kF32InBuffer = 0,
        kF32OutBuffer = 1,
    };
#define NUM_MACS    4

    typedef struct tagPqSmaxOpData{
        int bufNdc[2];
    }PqSmaxOpData_t;

    struct SoftmaxParamsExt {
      // beta is not really used (not a Tensorflow parameter) and not implemented
      // for LogSoftmax.
      double beta;
      // uint8_t inference params.  Used even when beta defaults to 1.0.
      int32_t input_multiplier;
      int32_t input_left_shift;
      // Reverse scaling is only used by LogSoftmax.
      int32_t reverse_scaling_divisor;
      int32_t reverse_scaling_right_shift;
      int diff_min;
      int32_t zero_point;
      float scale;
      float* table;
      // int16 LUT for exp(x), where x uniform distributed between [-10.0 , 0.0]
      int16_t* exp_lut;
      // int16 LUT for 1 / (1 + x), where x uniform distributed between [0.0 , 1.0]
      int16_t* one_over_one_plus_x_lut;
      uint8_t* uint8_table1;
      uint8_t* uint8_table2;
      // rocky:
      float inTensorScale;
      int bufNdc[2];
    };

// Softmax parameter data that persists in user_data
const int kInt16LUTArraySize = 513;

#if 0
TfLiteStatus CalculateSoftmaxParams(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    TfLiteTensor* output,
                                    const TfLiteSoftmaxParams* params,
                                    SoftmaxParams* op_data) {
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
      TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 32768,
                          (0.001f * 1.f / 32768));
    } else {  // input->type == kTfLiteInt8
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt8);
      if (output->type == kTfLiteInt16) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -32768);
        TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 65536,
                            (0.001f * 1.f / 65536));
      } else {  // output->type == kTfLiteint8
        TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
        // should use ENSURE_NEAR to tolerate small error
        // TF_LITE_ENSURE(context, output->params.scale == 1.f / 256);
        TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 256,
                            (0.001f * 1.f / 256));
      }
    }

    static const int kScaledDiffIntegerBits = 5;

    // Calculate input_multiplier and input_left_shift
    if (input->type == kTfLiteInt16) {
      int input_left_shift;
      double input_scale_beta_rescale =
          static_cast<double>(input->params.scale) *
          static_cast<double>(params->beta) /
          (10.0 / 65535.0);  // scale the input_diff such that [-65535, 0]
                             // correspond to [-10.0, 0.0]
      QuantizeMultiplier(input_scale_beta_rescale, &op_data->input_multiplier,
                         &input_left_shift);
      op_data->input_left_shift = input_left_shift;
    } else {
      int input_left_shift;
      tflite::PreprocessSoftmaxScaling(
          static_cast<double>(params->beta),
          static_cast<double>(input->params.scale), kScaledDiffIntegerBits,
          &op_data->input_multiplier, &input_left_shift);
      op_data->input_left_shift = input_left_shift;
      op_data->diff_min =
          -1.0 * tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                              op_data->input_left_shift);
    }
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
    op_data->beta = static_cast<double>(params->beta);
  }
  return kTfLiteOk;
}
#endif

void* PqSoftmaxInit(TfLiteContext* context, const char* buffer, size_t length) {
   PQ_Init(POWERQUAD);
   SCB->CPACR |= 3;

  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(SoftmaxParamsExt));
}

int smax_maxpool_by_neutron(const int8_t *pcLogits, int lenOfGroup) {
    int unitLen = lenOfGroup * NUM_MACS;
    int maxVal = -128;
    if (0)
    {
        // todo: really use Neutron to do
    }
    else {
        // simulation on CPU
        for (int i=0; i<unitLen; i++) {
            if (pcLogits[i] > maxVal)
                maxVal = pcLogits[i];
        }
    }
    return maxVal;
}

int smax_glbmaxpool(const int8_t* pcLogits, int unitLen) {
    int tailLen = unitLen % NUM_MACS;
    int alignedLen = unitLen - tailLen;
    int maxVal = smax_maxpool_by_neutron(pcLogits, alignedLen / NUM_MACS);
    const int8_t *pcTail = pcLogits + alignedLen;
    while (tailLen--) {
        if (pcTail[0] > maxVal) {
            maxVal = pcTail[0];
        }
        pcTail++;
    }
    return maxVal;

}

void pq_softmax_s8(TfLiteContext* context, const int8_t *pcLogits, const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    int8_t *pOut, int chnCnt, float scale, const int* pBufNdc)
{

    int i;
    float *pf32Iter;
    // subtract maxValue
    const int8_t *pcIterIn = pcLogits;
    int unitLen = num_rows * row_size;
    int maxVal = smax_glbmaxpool(pcLogits, unitLen);

    float *pf32Logits = (float*) context->GetScratchBuffer(context, pBufNdc[kF32InBuffer]);
    float *pf32Exp = (float*) context->GetScratchBuffer(context, pBufNdc[kF32OutBuffer]);
    // >>> calc exp with PQ

    float logit, act;

    float maxLogit = maxVal * scale;

    if (maxLogit >= 16.0f) {
        for (i=0, pf32Iter=pf32Logits; i<unitLen; i++) {
            logit = (float)(*pcIterIn++ - maxVal) * scale;
            *pf32Iter++ = logit;
        }
    } else {
        for (i=0, pf32Iter=pf32Logits; i<unitLen; i++) {
            logit = (float)(*pcIterIn++) * scale;
            *pf32Iter++ = logit;
        }
    }

    PQ_VectorEtoxF32(pf32Logits, pf32Exp, unitLen);

    int grpSize = unitLen / chnCnt;
    pf32Iter = pf32Exp;
    float *pf32Iter2 = pf32Iter;
    while(grpSize--) {
        float f32SmaxScale = 0;
        for (i=0; i<chnCnt; i++) {
            f32SmaxScale += *pf32Iter++;
        }
        f32SmaxScale = 1.0f / f32SmaxScale;

        for (i=0; i<chnCnt; i++) {
#if   defined ( __ICCARM__ )
            act = f32SmaxScale * *pf32Iter2++ * 256.0f - 128.0f;
            *pOut++ = __SSAT(MyRoundfNeareast(act), 8);
#else
            act = f32SmaxScale * *pf32Iter2++ * 256.0f - 128.0f + 0.5f;
            if (act > 0)
                *pOut++ = __SSAT((int)(act), 8); // sub 128, do not use "| 128", otherwise IAR will generate ORN #127
            else
                *pOut++ = __SSAT((int)(act - 1.0f), 8);
#endif
        }
    }

    // <<<
}

void SoftmaxInt8(TfLiteContext* context, const TfLiteEvalTensor* input, TfLiteEvalTensor* output,
                      int chnCnt, const SoftmaxParamsExt *op_data) {
      const auto input_shape = tflite::micro::GetTensorShape(input);
      const auto output_shape = tflite::micro::GetTensorShape(output);
      const int trailing_dim = input_shape.DimensionsCount() - 1;
      const int outer_size =
          MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
      const int depth =
          MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

#if 1
      pq_softmax_s8(context, tflite::micro::GetTensorData<int8_t>(input), outer_size,
                     depth, op_data->input_multiplier, op_data->input_left_shift,
                     op_data->diff_min, tflite::micro::GetTensorData<int8_t>(output),
                     chnCnt, op_data->inTensorScale, op_data->bufNdc);
#else
      arm_softmax_s8(tflite::micro::GetTensorData<int8_t>(input), outer_size,
                     depth, op_data->input_multiplier, op_data->input_left_shift,
                     op_data->diff_min,
                     tflite::micro::GetTensorData<int8_t>(output));
#endif
}

TfLiteStatus PqSoftMaxPrepare(TfLiteContext* context, TfLiteNode* node)
{
    auto* op_data = static_cast<SoftmaxParamsExt*>(node->user_data);

    TfLiteStatus status = SoftmaxPrepare(context, node);

    tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
    TfLiteTensor* pInput =
        micro_context->AllocateTempInputTensor(node, 0);
    op_data->inTensorScale = pInput->params.scale;
    int flatSize = GetTensorShape(pInput).FlatSize();
    size_t cbSize = (int) flatSize;

    // request scratch buffer for f32 form input and output (exp(x))
    context->RequestScratchBufferInArena(context, cbSize * sizeof(float), op_data->bufNdc + kF32InBuffer);
    context->RequestScratchBufferInArena(context, cbSize * sizeof(float), op_data->bufNdc + kF32OutBuffer);

    micro_context->DeallocateTempTfLiteTensor(pInput);

    return status;
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  switch (input->type) {
    case kTfLiteFloat32:
    {
      const SoftmaxParams data = *static_cast<const SoftmaxParams*>(node->user_data);
      tflite::reference_ops::Softmax(
          data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8:
    {
      auto op_data = static_cast<SoftmaxParamsExt*>(node->user_data);
      auto shp = tflite::micro::GetTensorShape(input);
      int chnDimNdx = shp.DimensionsCount() - 1;
      int chnCnt = input->dims->data[chnDimNdx];
      SoftmaxInt8(context, input, output, chnCnt, op_data);
      return kTfLiteOk;
    }
  case kTfLiteInt16:
      {
          const SoftmaxParams data = *static_cast<const SoftmaxParams*>(node->user_data);
          tflite::reference_ops::Softmax(
              data, tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
}

}  // namespace



TFLMRegistration Register_SOFTMAX() {
  return {/*init=*/PqSoftmaxInit,
          /*free=*/nullptr,
          /*prepare=*/PqSoftMaxPrepare,
          /*invoke=*/SoftmaxEval,
          /*reset=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr};
}

}  // namespace tflite
