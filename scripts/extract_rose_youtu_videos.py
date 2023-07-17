"""
Extracting faces from RoseYoutu dataset videos.

MTCNN Face Detection
Affine transformation for landmark alignment
Image Quality check (Laplacian variance)
Re-predict MTCNN on cropped

Batched MTCNN Predictions for speedup
"""

# stdlib
from glob import glob
import os
import sys
from os.path import join, basename
import argparse

# fix for local import problems
cwd = os.getcwd()
sys.path.extend([cwd] + [join(cwd, d) for d in os.listdir() if os.path.isdir(d)])

# external
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# local
from util_torch import init_device
from util_face import get_ref_landmarks, get_align_transform, transform_bbox, rotate, mtcnn_predict_batched, \
    find_rotation
from util_image import plot_many

# ashamed to put this here, but len(video) doesn't work on metacentrum
video_lengths = [492, 402, 427, 348, 350, 351, 347, 357, 381, 347, 366, 501, 401, 501, 382, 376, 332, 313, 398, 413,
                 393, 367, 357, 338, 328, 331, 342, 341, 360, 402, 356, 468, 383, 312, 344, 350, 339, 342, 148, 249,
                 159, 270, 233, 244, 194, 192, 192, 291, 180, 141, 161, 219, 221, 149, 494, 335, 187, 212, 301, 206,
                 204, 169, 325, 300, 278, 270, 174, 195, 179, 205, 181, 185, 276, 187, 222, 168, 181, 181, 203, 177,
                 164, 181, 164, 181, 179, 183, 179, 198, 219, 149, 185, 181, 175, 199, 205, 208, 443, 342, 534, 336,
                 376, 310, 317, 378, 340, 454, 299, 351, 584, 364, 304, 406, 173, 172, 165, 203, 175, 270, 257, 250,
                 202, 290, 185, 213, 167, 155, 280, 268, 269, 277, 235, 192, 254, 146, 260, 184, 321, 268, 176, 178,
                 178, 185, 306, 164, 403, 180, 183, 186, 247, 201, 258, 221, 174, 170, 169, 174, 159, 155, 228, 243,
                 131, 123, 164, 305, 275, 283, 139, 166, 307, 312, 340, 328, 359, 344, 325, 326, 321, 311, 328, 325,
                 312, 331, 343, 306, 314, 312, 313, 305, 310, 312, 289, 300, 298, 223, 235, 265, 282, 276, 489, 246,
                 218, 307, 377, 180, 198, 183, 294, 274, 419, 419, 343, 383, 379, 390, 349, 352, 424, 338, 363, 364,
                 342, 369, 387, 335, 406, 354, 361, 390, 354, 330, 342, 341, 375, 384, 397, 377, 344, 509, 496, 351,
                 366, 366, 426, 319, 333, 325, 325, 580, 596, 427, 466, 443, 423, 424, 417, 404, 399, 371, 359, 328,
                 454, 372, 374, 381, 379, 379, 390, 371, 350, 383, 347, 462, 388, 338, 344, 343, 409, 367, 478, 350,
                 354, 338, 376, 365, 372, 402, 151, 263, 219, 153, 250, 244, 235, 246, 189, 240, 246, 258, 265, 131,
                 138, 128, 201, 175, 184, 335, 311, 312, 386, 350, 287, 355, 248, 399, 289, 347, 300, 457, 399, 333,
                 269, 310, 307, 307, 236, 361, 383, 372, 306, 338, 291, 339, 305, 334, 132, 346, 313, 277, 186, 194,
                 221, 152, 188, 180, 268, 283, 206, 279, 270, 249, 189, 174, 334, 269, 220, 168, 348, 353, 340, 450,
                 393, 391, 382, 402, 362, 352, 330, 430, 355, 368, 423, 371, 404, 360, 486, 490, 369, 363, 428, 376,
                 345, 362, 358, 356, 339, 330, 351, 355, 406, 412, 432, 440, 406, 414, 426, 422, 391, 407, 389, 411,
                 352, 337, 381, 499, 384, 358, 456, 348, 377, 396, 388, 377, 214, 235, 183, 219, 179, 191, 171, 344,
                 163, 347, 159, 232, 168, 171, 165, 340, 323, 332, 340, 325, 304, 335, 301, 317, 352, 433, 333, 328,
                 303, 382, 305, 178, 178, 172, 170, 161, 183, 162, 158, 288, 196, 226, 190, 177, 163, 154, 149, 149,
                 149, 492, 337, 331, 334, 323, 352, 387, 393, 338, 268, 383, 322, 409, 324, 373, 331, 440, 372, 370,
                 327, 364, 332, 324, 310, 299, 424, 334, 386, 298, 363, 308, 396, 435, 391, 465, 364, 352, 359, 368,
                 337, 329, 342, 330, 364, 330, 441, 427, 344, 348, 336, 355, 326, 334, 375, 367, 494, 595, 338, 359,
                 459, 326, 358, 376, 418, 467, 165, 166, 164, 166, 195, 189, 188, 176, 174, 216, 203, 191, 180, 182,
                 333, 303, 330, 186, 159, 160, 157, 183, 161, 175, 227, 199, 177, 181, 182, 175, 160, 150, 147, 159,
                 317, 318, 227, 242, 199, 205, 193, 239, 202, 204, 197, 194, 189, 198, 191, 284, 191, 178, 187, 173,
                 331, 160, 165, 156, 186, 207, 208, 212, 213, 187, 159, 333, 312, 331, 160, 163, 155, 150, 149, 154,
                 163, 152, 184, 202, 184, 171, 207, 184, 172, 198, 165, 176, 186, 294, 207, 296, 298, 300, 296, 320,
                 293, 299, 323, 292, 318, 393, 340, 297, 309, 301, 333, 345, 339, 340, 374, 435, 420, 338, 348, 347,
                 434, 382, 336, 377, 394, 405, 415, 447, 451, 463, 338, 321, 338, 352, 355, 324, 338, 359, 354, 359,
                 359, 400, 307, 353, 372, 314, 325, 368, 336, 333, 348, 371, 352, 330, 326, 345, 346, 383, 356, 318,
                 328, 345, 438, 378, 335, 345, 340, 316, 320, 443, 396, 361, 176, 243, 219, 218, 190, 198, 171, 319,
                 338, 385, 185, 172, 320, 400, 421, 357, 345, 350, 386, 346, 334, 362, 407, 374, 342, 372, 347, 424,
                 343, 359, 360, 392, 371, 349, 357, 357, 143, 139, 149, 143, 144, 155, 147, 162, 147, 162, 201, 176,
                 155, 154, 147, 146, 152, 160, 175, 163, 164, 159, 173, 163, 175, 200, 177, 175, 158, 185, 204, 160,
                 191, 382, 195, 203, 203, 185, 207, 162, 163, 418, 359, 370, 361, 343, 349, 342, 344, 329, 367, 333,
                 415, 420, 451, 383, 349, 338, 366, 354, 366, 338, 346, 349, 379, 342, 382, 361, 336, 392, 378, 367,
                 418, 404, 390, 378, 371, 405, 460, 417, 459, 158, 169, 169, 340, 324, 328, 343, 328, 334, 342, 315,
                 345, 341, 357, 329, 324, 338, 334, 329, 343, 341, 342, 347, 325, 325, 467, 394, 139, 150, 240, 241,
                 144, 154, 159, 234, 292, 146, 250, 251, 163, 281, 328, 157, 142, 148, 226, 206, 385, 358, 359, 364,
                 400, 354, 425, 380, 361, 399, 404, 357, 390, 425, 359, 327, 382, 410, 420, 153, 294, 536, 339, 346,
                 345, 325, 317, 360, 334, 322, 354, 362, 359, 339, 356, 371, 336, 377, 335, 317, 325, 154, 154, 169,
                 166, 158, 144, 144, 186, 163, 173, 166, 157, 139, 162, 164, 151, 159, 164, 192, 167, 145, 140, 171,
                 210, 173, 211, 203, 174, 197, 176, 235, 159, 180, 184, 156, 134, 186, 179, 238, 166, 171, 214, 184,
                 191, 190, 191, 301, 296, 203, 215, 217, 199, 254, 155, 165, 206, 175, 322, 324, 324, 338, 322, 318,
                 323, 344, 362, 322, 365, 329, 348, 366, 508, 341, 325, 332, 333, 282, 310, 343, 479, 351, 317, 322,
                 334, 380, 382, 319, 328, 436, 361, 398, 356, 397, 358, 379, 372, 366, 383, 376, 369, 318, 369, 311,
                 374, 346, 330, 345, 322, 348, 347, 352, 329, 321, 320, 343, 387, 346, 326, 317, 346, 125, 189, 191,
                 352, 257, 209, 299, 193, 294, 384, 202, 218, 197, 217, 151, 128, 222, 230, 259, 142, 210, 205, 242,
                 220, 214, 194, 196, 172, 188, 195, 258, 250, 211, 560, 358, 342, 356, 362, 374, 441, 365, 409, 356,
                 388, 425, 359, 388, 428, 387, 302, 318, 398, 351, 357, 222, 381, 622, 163, 395, 190, 389, 339, 420,
                 320, 342, 375, 375, 359, 452, 408, 374, 320, 353, 399, 342, 343, 355, 341, 340, 344, 352, 403, 391,
                 352, 394, 378, 485, 419, 443, 393, 342, 351, 329, 381, 354, 329, 339, 360, 481, 382, 504, 400, 325,
                 549, 348, 475, 362, 338, 339, 335, 346, 362, 339, 374, 361, 427, 496, 368, 397, 345, 372, 380, 357,
                 337, 402, 457, 455, 374, 371, 394, 379, 445, 331, 368, 357, 370, 362, 349, 342, 374, 339, 353, 357,
                 370, 378, 434, 393, 327, 434, 364, 384, 575, 492, 518, 435, 566, 432, 383, 418, 432, 475, 625, 541,
                 339, 335, 442, 387, 343, 471, 448, 421, 466, 374, 325, 428, 418, 437, 398, 368, 397, 420, 382, 417,
                 448, 378, 372, 286, 369, 452, 327, 443, 332, 444, 420, 427, 471, 398, 472, 348, 338, 300, 349, 323,
                 326, 388, 321, 148, 157, 164, 154, 163, 165, 154, 166, 194, 169, 183, 166, 164, 171, 160, 175, 160,
                 351, 180, 148, 263, 144, 250, 245, 245, 279, 300, 176, 199, 253, 228, 180, 176, 185, 185, 193, 272,
                 247, 245, 245, 272, 234, 160, 283, 340, 364, 395, 368, 486, 359, 440, 419, 514, 364, 471, 439, 425,
                 343, 320, 309, 362, 341, 160, 160, 153, 155, 166, 200, 166, 149, 182, 176, 179, 184, 231, 173, 172,
                 172, 192, 203, 175, 175, 158, 147, 154, 146, 197, 150, 151, 154, 144, 157, 173, 151, 161, 161, 158,
                 147, 159, 163, 161, 156, 156, 146, 156, 158, 212, 163, 162, 141, 155, 337, 335, 332, 321, 434, 386,
                 365, 380, 349, 335, 371, 341, 336, 338, 322, 334, 350, 401, 334, 373, 293, 322, 295, 294, 294, 348,
                 293, 327, 293, 361, 337, 299, 345, 326, 303, 307, 292, 345, 340, 344, 502, 351, 387, 174, 155, 135,
                 159, 164, 157, 185, 182, 193, 160, 340, 380, 178, 202, 209, 183, 148, 222, 137, 159, 153, 180, 181,
                 173, 157, 176, 138, 129, 157, 151, 176, 169, 181, 153, 152, 194, 178, 173, 183, 165, 183, 171, 162,
                 176, 164, 167, 168, 358, 357, 203, 245, 324, 318, 330, 146, 161, 390, 161, 142, 329, 339, 270, 195,
                 187, 260, 341, 136, 215, 284, 358, 152, 148, 147, 154, 157, 146, 171, 149, 160, 142, 291, 297, 306,
                 312, 300, 302, 295, 310, 293, 167, 159, 163, 155, 155, 123, 130, 158, 168, 174, 191, 122, 165, 144,
                 143, 187, 336, 325, 320, 340, 365, 409, 337, 375, 360, 357, 362, 346, 413, 349, 325, 330, 352, 344,
                 372, 155, 324, 345, 313, 336, 328, 318, 325, 338, 370, 338, 343, 368, 334, 334, 319, 357, 357, 342,
                 411, 142, 160, 397, 383, 342, 404, 406, 328, 375, 333, 390, 334, 353, 361, 345, 199, 186, 329, 326,
                 341, 349, 348, 322, 323, 381, 343, 342, 334, 369, 317, 338, 354, 316, 330, 355, 322, 316, 313, 334,
                 385, 358, 334, 325, 322, 327, 347, 334, 370, 380, 379, 337, 318, 332, 397, 340, 311, 292, 308, 292,
                 302, 339, 419, 413, 377, 445, 418, 314, 298, 411, 408, 384, 423, 355, 442, 561, 400, 344, 416, 441,
                 518, 333, 335, 414, 376, 373, 423, 383, 365, 445, 541, 421, 472, 392, 477, 401, 350, 418, 379, 435,
                 354, 335, 319, 396, 406, 162, 164, 166, 328, 324, 349, 310, 371, 307, 292, 320, 310, 348, 323, 324,
                 322, 326, 326, 341, 322, 326, 317, 334, 314, 314, 344, 321, 318, 377, 300, 340, 338, 384, 336, 351,
                 321, 351, 320, 324, 347, 331, 322, 339, 375, 321, 299, 303, 322, 318, 169, 150, 165, 189, 144, 156,
                 136, 156, 153, 147, 165, 166, 108, 154, 186, 153, 136, 167, 172, 314, 339, 345, 318, 422, 378, 334,
                 322, 289, 357, 331, 300, 279, 312, 285, 259, 337, 418, 383, 360, 441, 569, 293, 315, 292, 358, 324,
                 336, 126, 121, 174, 179, 165, 171, 244, 163, 167, 162, 135, 130, 381, 335, 348, 381, 409, 363, 333,
                 345, 351, 321, 329, 329, 340, 316, 341, 155, 170, 158, 349, 355, 181, 320, 340, 318, 335, 390, 352,
                 315, 413, 164, 347, 330, 361, 370, 346, 343, 391, 369, 346, 354, 373, 384, 357, 332, 328, 351, 363,
                 399, 376, 334, 170, 123, 166, 164, 144, 121, 142, 166, 162, 171, 174, 142, 162, 169, 165, 166, 138,
                 140, 129, 139, 161, 369, 384, 346, 365, 333, 351, 373, 341, 372, 346, 377, 331, 328, 346, 369, 340,
                 163, 140, 154, 158, 163, 159, 163, 317, 320, 318, 329, 345, 436, 354, 335, 334, 314, 339, 317, 348,
                 389, 314, 313, 330, 318, 315, 345, 346, 335, 324, 359, 345, 334, 337, 340, 353, 315, 318, 331, 318,
                 317, 353, 324, 348, 327, 355, 349, 297, 337, 307, 317, 325, 326, 188, 181, 162, 224, 154, 162, 193,
                 158, 177, 185, 161, 205, 195, 163, 178, 181, 177, 175, 166, 344, 377, 418, 337, 363, 332, 380, 383,
                 351, 337, 359, 329, 350, 338, 357, 352, 353, 394, 343, 335, 339, 383, 346, 350, 356, 378, 339, 346,
                 281, 304, 301, 274, 303, 151, 146, 133, 144, 148, 141, 145, 104, 171, 197, 160, 132, 146, 132, 106,
                 103, 133, 128, 154, 161, 174, 152, 167, 156, 162, 152, 156, 155, 168, 163, 152, 151, 152, 172, 153,
                 153, 160, 161, 163, 164, 158, 156, 158, 155, 119, 167, 171, 160, 156, 167, 225, 158, 147, 154, 156,
                 147, 156, 161, 385, 347, 348, 356, 362, 370, 395, 381, 346, 381, 344, 360, 444, 311, 409, 378, 344,
                 442, 349, 337, 357, 362, 138, 174, 169, 161, 140, 156, 217, 171, 171, 133, 132, 123, 123, 173, 152,
                 184, 142, 152, 132, 154, 163, 156, 137, 127, 146, 287, 293, 139, 145, 318, 341, 255, 299, 280, 292,
                 309, 328, 335, 323, 415, 314, 321, 331, 321, 315, 301, 318, 322, 314, 310, 316, 305, 314, 307, 339,
                 319, 328, 338, 172, 156, 377, 358, 445, 161, 146, 160, 159, 315, 350, 320, 344, 305, 343, 317, 413,
                 375, 373, 308, 370, 321, 323, 495, 465, 423, 419, 474, 555, 292, 394, 324, 449, 335, 368, 382, 528,
                 387, 139, 255, 224, 228, 222, 228, 232, 254, 233, 222, 216, 131, 152, 131, 131, 151, 220, 152, 222,
                 148, 148, 265, 245, 149, 146, 236, 259, 144, 153, 161, 156, 157, 154, 157, 146, 159, 158, 159, 160,
                 164, 156, 164, 157, 155, 168, 159, 153, 160, 151, 158, 151, 157, 149, 317, 311, 342, 314, 338, 314,
                 314, 317, 334, 324, 328, 326, 320, 328, 325, 318, 324, 322, 325, 309, 314, 338, 318, 314, 320, 325,
                 321, 323, 322, 321, 352, 341, 346, 363, 336, 349, 338, 451, 307, 322, 327, 317, 307, 310, 338, 334,
                 268, 133, 229, 233, 251, 238, 330, 353, 335, 347, 328, 318, 323, 331, 320, 314, 344, 365, 316, 338,
                 325, 330, 359, 385, 371, 359, 340, 323, 339, 387, 333, 328, 312, 320, 329, 321, 351, 348, 367, 358,
                 372, 346, 392, 339, 358, 350, 340, 346, 349, 354, 323, 364, 340, 354, 338, 341, 177, 136, 162, 141,
                 111, 141, 193, 152, 165, 172, 134, 160, 170, 166, 162, 172, 163, 158, 162, 153, 157, 166, 157, 183,
                 150, 154, 151, 167, 156, 163, 182, 171, 161, 165, 164, 155, 151, 171, 161, 329, 328, 326, 340, 364,
                 348, 336, 332, 327, 341, 312, 333, 340, 338, 313, 332, 310, 315, 310, 306, 230, 132, 247, 226, 246,
                 238, 223, 222, 127, 234, 127, 128, 153, 221, 216, 220, 149, 155, 159, 169, 160, 157, 164, 162, 148,
                 156, 167, 161, 167, 158, 152, 147, 168, 159, 153, 158, 157, 157, 159, 152, 318, 295, 297, 323, 312,
                 269, 279, 298, 305, 258, 269, 302, 309, 305, 308, 220, 149, 236, 125, 152, 128, 232, 234, 246, 127,
                 162, 228, 133, 250, 233, 234, 228, 147, 269, 160, 226, 225, 218, 216, 221, 132, 145, 132, 122, 110,
                 148, 135, 127, 229, 172, 233, 145, 156, 143, 155, 141, 150, 146, 143, 145, 369, 338, 311, 308, 342,
                 319, 343, 153, 133, 219, 330, 316, 347, 303, 158, 160, 157, 155, 286, 316, 284, 316, 352, 276, 306,
                 323, 309, 359, 342, 284, 246, 350, 241, 346, 406, 417, 348, 381, 409, 497, 278, 303, 322, 139, 144,
                 173, 220, 145, 140, 267, 129, 290, 143, 128, 143, 292, 133, 281, 328, 464, 331, 323, 280, 290, 266,
                 312, 300, 310, 276, 275, 284, 256, 286, 358, 336, 364, 34, 390, 358, 321, 306, 328, 307, 310, 314, 320,
                 323, 304, 322, 325, 317, 323, 334, 311, 310, 319, 286, 382, 294, 264, 303, 263, 340, 288, 292, 291,
                 333, 287, 350, 317, 293, 325, 281, 261, 301, 389, 319, 319, 318, 289, 276, 155, 150, 197, 212, 217,
                 133, 122, 148, 192, 166, 156, 156, 161, 147, 139, 174, 190, 201, 201, 197, 174, 144, 207, 146, 144,
                 144, 144, 166, 165, 159, 185, 145, 142, 162, 160, 142, 142, 165, 162, 170, 178, 149, 168, 157, 161,
                 154, 154, 158, 157, 158, 158, 156, 162, 147, 136, 161, 162, 153, 134, 143, 103, 98, 267, 224, 153, 139,
                 219, 269, 141, 232, 139, 157, 127, 156, 148, 216, 188, 132, 332, 346, 340, 297, 308, 352, 277, 318,
                 396, 349, 358, 325, 331, 285, 269, 317, 322, 311, 325, 326, 310, 286, 310, 389, 314, 341, 314, 405,
                 286, 294, 307, 287, 314, 295, 288, 322, 325, 296, 319, 384, 267, 280, 269, 375, 290, 375, 402, 303,
                 336, 279, 300, 310, 361, 347, 304, 303, 325, 315, 250, 580, 442, 416, 357, 343, 331, 347, 388, 379,
                 341, 353, 642, 346, 353, 340, 349, 341, 327, 330, 339, 337, 318, 344, 287, 141, 138, 146, 144, 145,
                 322, 147, 170, 144, 171, 182, 141, 311, 345, 347, 367, 437, 342, 258, 279, 290, 281, 334, 297, 327,
                 317, 357, 337, 282, 406, 383, 252, 343, 282, 346, 311, 318, 300, 306, 334, 320, 279, 276, 312, 318,
                 277, 301, 279, 271, 241, 546, 388, 370, 319, 456, 584, 299, 305, 300, 304, 278, 127, 119, 142, 323,
                 377, 275, 317, 142, 138, 150, 290, 390, 329, 396, 144, 179, 146, 231, 145, 116, 172, 130, 144, 122,
                 405, 301, 173, 141, 139, 148, 133, 137, 156, 207, 189, 146, 140, 145, 284, 407, 417, 277, 385, 328,
                 323, 306, 407, 404, 314, 355, 348, 335, 326, 547, 450, 412, 377, 480, 541, 330, 353, 378, 360, 173,
                 173, 143, 190, 152, 143, 179, 148, 176, 168, 170, 285, 302, 309, 305, 280, 383, 332, 299, 306, 303,
                 304, 255, 308, 337, 359, 326, 312, 296, 304, 345, 356, 307, 338, 445, 392, 438, 544, 414, 372, 362,
                 363, 367, 362, 366, 410, 416, 370, 417, 332, 486, 337, 391, 350, 325, 133, 144, 152, 121, 133, 143,
                 127, 130, 148, 122, 160, 149, 134, 183, 125, 150, 134, 132, 191, 161, 176, 152, 150, 166, 135, 140,
                 155, 189, 179, 162, 166, 150, 118, 129, 168, 178, 150, 314, 435, 350, 322, 347, 463, 392, 385, 354,
                 322, 370, 307, 309, 363, 442, 420, 402, 406, 374, 301, 317, 353, 336, 462, 577, 313, 317, 336, 372,
                 310, 313, 405, 487, 407, 310, 274, 313, 311, 305, 339, 327, 308, 281, 311, 396, 372, 351, 255, 301,
                 222, 375, 286, 273, 383, 478, 344, 124, 133, 262, 174, 160, 132, 256, 295, 136, 136, 263, 177, 403,
                 178, 437, 271, 140, 143, 363, 337, 372, 383, 376, 364, 340, 323, 252, 323, 310, 700, 628, 489, 493,
                 486, 484, 438, 390, 427, 427, 560, 265, 247, 270, 281, 437, 315, 286, 233, 206, 141, 153, 274, 148,
                 161, 216, 232, 136, 311, 338, 347, 377, 367, 198, 321, 331, 318, 325, 320, 319, 318, 325, 321, 324,
                 321, 320, 320, 327, 205, 306, 343, 319, 272, 372, 323, 301, 274, 314, 268, 282, 472, 433, 381, 368,
                 446, 589, 286, 364, 310, 337, 346, 324, 371, 348, 386, 360, 323, 279, 349, 319, 451, 426, 254, 357,
                 289, 343, 281, 308, 298, 324, 268, 280, 375, 298, 311, 413, 257, 346, 335, 268, 426, 391, 339, 384,
                 459, 380, 295, 301, 300, 347, 370, 346, 366, 286, 285, 343, 427, 384, 393, 370, 380, 328, 337, 330,
                 327, 300, 309, 350, 338, 352, 326, 338, 365, 416, 396, 425, 403, 317, 353, 643, 664, 521, 529, 465,
                 438, 424, 418, 421, 412, 591, 523, 366, 637, 386, 352, 349, 380, 381, 384, 375, 400, 348, 380, 372,
                 407, 430, 404, 323, 306, 400, 118, 150, 162, 147, 190, 191, 159, 185, 137, 169, 136, 139, 204, 253,
                 203, 190, 208, 230, 130, 161, 138, 347, 332, 310, 378, 392, 367, 420, 358, 414, 371, 373, 372, 366,
                 377, 365, 294, 287, 323, 312, 609, 604, 489, 455, 442, 431, 415, 417, 395, 391, 447, 434, 351, 400,
                 389, 374, 427, 323, 330, 271, 296, 348, 342, 311, 372, 462, 373, 364, 429, 247, 344, 323, 309, 301,
                 338, 463, 572, 668, 491, 481, 405, 444, 428, 430, 414, 416, 473, 430, 354, 399, 470, 367, 348, 364,
                 269, 359, 489, 441, 380, 375, 330, 330, 320, 326, 333, 530, 351, 356, 367, 335, 335, 296, 337, 325,
                 340, 325, 510, 415, 417, 353, 386, 389, 338, 338, 328, 352, 473, 395, 356, 361, 349, 326, 310, 295,
                 309, 367, 336, 325, 369, 352, 408, 346, 368, 504, 383, 360, 368, 371, 363, 360, 347, 362, 343, 373,
                 422, 403, 389, 478, 317, 306, 305, 289, 309, 289, 429, 151, 322, 331, 303, 293, 286, 384, 330, 336,
                 300, 243, 373, 314, 210, 168, 133, 165, 353, 260, 391, 292, 282, 280, 274, 316, 192, 197, 204, 239,
                 331, 161, 181, 164, 179, 167, 161, 165, 157, 188, 161, 136, 184, 252, 435, 657, 335, 358, 342, 342,
                 380, 319, 415, 363, 395, 385, 386, 337, 372, 436, 333, 307, 382, 282, 302, 222, 169, 142, 209, 378,
                 366, 339, 404, 366, 353, 338, 437, 415, 349, 333, 339, 271, 337, 344, 382, 380, 367, 337, 380, 417,
                 368, 380, 376, 826, 346, 343, 400, 393, 399, 176, 238, 271, 256, 248, 195, 169, 194, 198, 225, 239,
                 181, 181, 184, 208, 190, 176, 231, 215, 196, 157, 153, 160, 323, 329, 321, 29, 355, 320, 312, 309, 327,
                 328, 311, 306, 314, 316, 331, 294, 296, 344, 472, 406, 260, 283, 290, 277, 318, 291, 308, 309, 308,
                 320, 297, 321, 386, 408, 395, 276, 270, 240, 270, 286, 270, 309, 370]

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', help='directory with videos', type=str,
                    required=True)  # e.g. '/mnt/sdb1/dp/rose_youtu_vids'
parser.add_argument('-o', '--output_path', help='output directory, must not exist', type=str,
                    required=True)  # e.g. '/mnt/sdb1/dp/rose_youtu_imgs'
parser.add_argument('-f', '--frames', type=int, help='number of frames to extract per video',
                    required=True)  # e.g. 100
parser.add_argument('-b', '--batch_size', type=int, help='batch size', required=True)  # e.g. 64

''' Global Variables '''
EXT = '.jpg'
mtcnn = None  # face detector

img_size = np.array([384, 384])  # size of images to save
SCALE = 3.25
MARGIN = 384 - 112 * SCALE
ref_pts = get_ref_landmarks(scale=SCALE, margin=MARGIN)  # IPT reference landmarks (5) for face alignment
TH_LAPLACIAN = 20  # empirical threshold for laplacian variance, blurry images, discard if below

output_path = None
metadata = None  # dataframe for metadata, saved as output
total_samples = 0


def process_batch(batch, crop_names):
    """ Process a batch of images, saving them to disk.

    :param batch: batch of images
    :param crop_names: filenames for cropped images
    :return: number of images saved
    box format: [x0, y0, x1, y1]
    landmark format: [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4] =
                    [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    global mtcnn, metadata, output_path, ref_pts, img_size
    n_saved = 0
    try:
        ''' Batch-predict '''
        boxes, probs, landmarks = mtcnn_predict_batched(mtcnn, batch)
        ''' Keep only detected faces frames '''
        valid_idxs = np.array([i for i, b in enumerate(boxes) if b is not None])
        # ^^^ forget about np.where, mtcnn.detect returns [None] when len(batch) == 1 and None when >= 2
        boxes, probs, landmarks, crop_names = boxes[valid_idxs], probs[valid_idxs], landmarks[valid_idxs], crop_names[
            valid_idxs]
        batch = [batch[i] for i in valid_idxs]
        ''' Keep best bounding box per frame '''
        boxes, probs, landmarks = mtcnn.select_boxes(boxes, probs, landmarks, batch, method=mtcnn.selection_method)
        frame_dims = np.array([b.size for b in batch])
        boxes, landmarks = boxes[:, 0, :], landmarks[:, 0, :]  # remove 1-dim
        # check if bounding box is inside the frame
        valid_idxs = []
        for i, b in enumerate(boxes):  # check for bbox outside the frame
            if b[0] >= 0 and b[1] >= 0 and b[2] <= frame_dims[i][0] and b[3] <= frame_dims[i][1]:
                valid_idxs.append(i)

        # filter valid idxs again
        batch = [batch[i] for i in valid_idxs]
        boxes, probs, landmarks, crop_names = boxes[valid_idxs], probs[valid_idxs], landmarks[valid_idxs], crop_names[
            valid_idxs]
        frame_dims = [frame_dims[i] for i in valid_idxs]

        ''' Per-image saving of cropped faces, and metadata '''
        crops = []
        crops_data = []
        for i, box in enumerate(boxes):
            ''' Align face'''
            transform = get_align_transform(landmarks[i], ref_pts)
            crop = cv2.warpAffine(np.array(batch[i]), transform, img_size, borderMode=cv2.BORDER_CONSTANT,
                                  flags=cv2.INTER_AREA)
            box_cropped = transform_bbox(boxes[i], transform)
            ''' Check for low-quality crops '''
            laplacian_var = cv2.Laplacian(crop, cv2.CV_64F).var()
            if laplacian_var < TH_LAPLACIAN:
                continue  # skip blurry images

            crops.append(crop)
            crops_data.append({'path': crop_names[i],  # save metadata
                               'box_orig': np.int32(boxes[i]),  # frame-coords
                               'box': box_cropped,  # crop-coords
                               'landmarks': np.int32(landmarks[i]),  # frame-coords
                               'dim_orig': frame_dims[i],  # frame-coords
                               'face_prob': probs[i],
                               'laplacian_var': laplacian_var})
        ''' Re-predict on cropped images '''
        crops = np.array(crops)
        c_boxes, c_probs, c_landmarks = mtcnn_predict_batched(mtcnn, crops)
        # get valid indexes and apply them
        valid_idxs = np.array([i for i, b in enumerate(c_boxes) if b is not None])
        c_boxes, c_probs, c_landmarks = c_boxes[valid_idxs], c_probs[valid_idxs], c_landmarks[valid_idxs]
        crops_data = [crops_data[i] for i in valid_idxs]
        crops = [crops[i] for i in valid_idxs]
        c_boxes, c_probs, c_landmarks = mtcnn.select_boxes(c_boxes, c_probs, c_landmarks, crops,
                                                           method=mtcnn.selection_method)
        assert (len(c_boxes) == len(c_probs) == len(c_landmarks) == len(crops_data) == len(crops))
        for i, ls in enumerate(c_landmarks):
            if ls is None:
                continue
            ls = np.float32(ls[0])
            # Verify landmarks are close to the reference landmarks
            dst_ls = np.mean(np.linalg.norm(ls - ref_pts, axis=1))
            if dst_ls > 30:
                continue
            ''' Save image + metadata '''
            path_crop = join(output_path, crops_data[i]['path'])
            cv2.imwrite(path_crop, cv2.cvtColor(crops[i], cv2.COLOR_RGB2BGR))
            metadata.append(crops_data[i])
            n_saved += 1
    except Exception as e:
        # catch error when empty valid idxs, or another error, and just continue
        # happens in cases where no crops would have been valid anyway
        print('process_batch:', e)
    finally:
        return n_saved


# def main():
if __name__ == '__main__':
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))
    ''' Setup '''
    # Output directory
    output_path = args.output_path
    path_metadata = join(output_path, 'data.pkl')
    os.makedirs(output_path, exist_ok=False)
    print(f'Created dir: {output_path}')
    # Input videos paths
    input_path = args.input_path
    matching_pattern = join(input_path, '*.mp4')
    start = 0
    vid_paths = glob(matching_pattern)[start:]
    print(f'Found {len(vid_paths)} files matching "{matching_pattern}"')
    ''' Face Detector - MTCNN '''
    device = init_device()
    mtcnn = MTCNN(image_size=384, select_largest=True, device=device, post_process=False, min_face_size=150)
    # vid_filename = filenames[0]
    metadata = []
    err_counter = 0
    total_samples = 0
    rotations = []

    ''' Processing videos '''
    pbar = tqdm(enumerate(vid_paths, start=start), total=len(vid_paths))
    frames, v_names, crop_names = [], [], []
    for i_vid, vid_path in pbar:
        vid_filename = basename(vid_path)
        filename_base = vid_filename[: -len(".mp4")]
        samples_saved = 0
        try:
            video = cv2.VideoCapture(join(input_path, vid_filename))
            rot_meta = video.get(cv2.CAP_PROP_ORIENTATION_META)
            success, frame = video.read()  # testing frame
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rot = find_rotation(mtcnn, frame)
            rotations.append({'source': vid_filename, 'rotation': rot, 'rotation_meta': rot_meta})
            if rot is None:
                continue
            # v_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # does not work on metacentrum
            v_len = video_lengths[i_vid]
            ''' Process video  '''
            filter_rate = v_len // args.frames
            filter_rate = max(1, filter_rate)  # avoid zero-div
            for i_frame in range(1000):  # max video length = 826
                # Load frame
                success, frame = video.read()
                last_frame = not success and i_vid == len(vid_paths) - 1
                skipped = i_frame % filter_rate != 0

                # Process frame
                if not skipped and success:
                    frame = rotate(frame, rot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Accumulate batch
                    frames.append(Image.fromarray(frame))
                    crop_names.append(f'{filename_base}_crop_{i_frame}{EXT}')

                # Batch Predict
                if len(frames) >= args.batch_size or last_frame:
                    # ^ full batch, or last frame of video ^
                    samples_saved += process_batch(frames, np.array(crop_names))
                    frames, crop_names = [], []
                    pbar.set_description(desc=f'vid {i_vid}/{len(vid_paths)}, '
                                              f'frame {i_frame}/{v_len}, '
                                              f'saved {total_samples} (+{samples_saved})')
                if not success:
                    break
            # end of video
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            break
        except Exception as e:
            if err_counter < 5:
                print('main:', e)
            err_counter += 1
        finally:
            total_samples += samples_saved

    # end of all videos
    print(f'Error count: {err_counter} / {len(vid_paths)}')
    # count errors of videos, as per-frame problems don't bubble up here

    df = pd.DataFrame(metadata)
    df.to_pickle(path_metadata)
    print(f'Saved df to {path_metadata}')
    df_dup = df[df.duplicated(subset=['path'], keep=False)]  # non-unique path
    if len(df_dup) > 0:
        print(f'Duplicate paths: {len(df_dup)}')
        df.to_pickle(join(output_path, 'metadata_duplicates.pkl'))

    df_rot = pd.DataFrame(rotations)
    df_rot.to_pickle(join(output_path, 'metadata_rotations.pkl'))
    print(f'No rotation found: {sum(df_rot.rotation.isna())}')

    # find videos without samples
    videos_with_samples = df.path.apply(lambda x: x.split('_crop_')[0]).unique()
    vids_basenames = set(basename(p)[:-len(".mp4")] for p in vid_paths)
    no_samples_vids = vids_basenames - set(videos_with_samples)

    print(f'No sample vids: {len(no_samples_vids)}')
    # save no-sample vids
    with open(join(output_path, 'metadata_vids_skipped.txt'), 'w') as f:
        f.write('\n'.join(no_samples_vids))
