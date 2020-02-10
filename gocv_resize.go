package main

import (
	"fmt"
	"image"
	"log"
	"os"

	"gocv.io/x/gocv"
)

func main() {

	// notFoodImgPath := "/Users/kahlil/Pictures/plate-food.jpg"
	// foodImgPath := "/Users/kahlil/Documents/AIA AI Food Scoring /Food NotFood testing/Test1 /Images for testing/8453869e-d8ce-4009-8011-8a20b58b2037-large-thumb.jpg"
	// plateFoodImgPath := "/Users/kahlil/Documents/AIA AI Food Scoring /Food NotFood testing/S3 rated imgs/1.0/2.jpg"
	// s3FoodImgPath := "/Users/kahlil/Documents/AIA AI Food Scoring /Food NotFood testing/S3 rated imgs/1.0/2.jpg"

	// NOTE : It is VERY important that the folder(dir) path is given with a trailing slash
	// If this is ever going to be anywhere near production then
	// we need to find a better way to add the file name to the directory (folder) path
	foodImgsFolderPath := os.Args[1]

	// filename := s3FoodImgPath

	file, openErr := os.Open(foodImgsFolderPath)
	if openErr != nil {
		log.Fatal(openErr)
	}
	imgFileNames, readNamesErr := file.Readdirnames(0)
	if readNamesErr != nil {
		log.Fatal(readNamesErr)
	}
	fmt.Println(imgFileNames)
	imgCount := len(imgFileNames)
	fmt.Println("Image count: ", imgCount)
	var allImgsPixelsArray [][3][224][224]float32

	for idx := 0; idx < imgCount; idx++ {
		currentImgFileName := foodImgsFolderPath + (imgFileNames[idx])
		// fmt.Printf("Current file name: %v\n", currentImgFileName)

		// for each image file, we need to get the file and then create the tensor and pass it to
		// our function that accepts the tensor (of all imgs) and loads and runs the model.

		img := gocv.IMRead(currentImgFileName, gocv.IMReadColor)
		if img.Empty() {
			fmt.Printf("Error reading image from: %v\n", currentImgFileName)
			return
		}

		resizedImg := gocv.NewMat()
		gocv.Resize(img, &resizedImg, image.Pt(224, 224), 0.0, 0.0, gocv.InterpolationCubic)

		imgPixels, errMsg := reshapeImgPixelsForTensor(resizedImg)
		if errMsg != nil {
			log.Fatal(errMsg)
		}
		// allImgsPixelsArray[idx] = imgPixels
		allImgsPixelsArray = append(allImgsPixelsArray, imgPixels)
	}
	runInferenceModel(imgFileNames, allImgsPixelsArray)

}

// func getImgFileNamesInFolder (){}

// func readAndResizeImage(){}

// func

func standardizeRed(inputPixelVal float32) (standardizedVal float32) {
	return (inputPixelVal - 0.485) / 0.229
}

func standardizeGreen(inputPixelVal float32) (standardizedVal float32) {
	return (inputPixelVal - 0.456) / 0.224
}

func standardizeBlue(inputPixelVal float32) (standardizedVal float32) {
	return (inputPixelVal - 0.406) / 0.225
}

func reshapeImgPixelsForTensor(resizedImg gocv.Mat) (imgForTensor [3][224][224]float32, err error) {

	var allChannelsImgPixels [3][224][224]float32

	height := resizedImg.Rows()
	// fmt.Println("Rows and Cols : {} {} ", height, width)
	step := resizedImg.Step()
	imgData := resizedImg.ToBytes()

	channels := resizedImg.Channels()
	var redPixels [224][224]float32
	var greenPixels [224][224]float32
	var bluePixels [224][224]float32

	for y := 0; y < height; y++ {
		// var row [][]uint8
		widthCtr := 0
		for x := 0; x < step; x = x + channels {
			B := standardizeBlue(float32(uint8(imgData[y*step+x])) / 255.0)
			G := standardizeGreen(float32(uint8(imgData[y*step+x+1])) / 255.0)
			R := standardizeRed(float32(uint8(imgData[y*step+x+2])) / 255.0)
			if channels == 4 {
				_ = uint8(imgData[y*step+x+3])
			}
			// row = append(row, {R,G,B} )
			// fmt.Println("Pixel values: ", B, G, R)
			// fmt.Println("The index vals (y,x)", y, x)
			widthCtr++
			// fmt.Println("Index val: ", y, widthCtr-1)
			redPixels[y][widthCtr-1] = R
			greenPixels[y][widthCtr-1] = G
			bluePixels[y][widthCtr-1] = B
		}
	}
	allChannelsImgPixels[0] = redPixels
	allChannelsImgPixels[1] = greenPixels
	allChannelsImgPixels[2] = bluePixels

	// To be used for debugging Pixel values of each channel for comparison with Rust & Python
	// for ctrVal := 0; ctrVal < 10; ctrVal++ {
	// 	fmt.Println("Red channel first 10 pixels :", redPixels[0][ctrVal])
	// }

	// fmt.Println()

	// for ctrVal := 0; ctrVal < 10; ctrVal++ {
	// 	fmt.Println("Green Channel first 10 pixels :", greenPixels[0][ctrVal])
	// }

	// fmt.Println()

	// for ctrVal := 0; ctrVal < 10; ctrVal++ {
	// 	fmt.Println("Blue Channel first 10 pixels :", bluePixels[0][ctrVal])
	// }

	return allChannelsImgPixels, nil
	// 	// var a [4]int

	// 	m := int(nrow * ncol)
	// 	imgs[i] = make(RawImage, m)
	// 	m_, err := io.ReadFull(r, imgs[i])
	// 	// fmt.Println("Value of m_ & image inside loop: ", m_, imgs[i])

	// 	currentImageBytes := imgs[i]
	// 	currentImageFloatVals := make([][]float32, m)
	// 	var newFmtImg [][][]float32

	// 	// imgBytesFloat := make([]float64, m)
	// 	// for ctr, element := range currentImageBytes {
	// 	// 	imgBytesFloat[ctr] = float64(element)
	// 	// }
	// 	// newFmtImgs := mat.NewDense(int(nrow), int(ncol), imgBytesFloat)
	// 	// fmt.Println("New Matrix created is: ", newFmtImgs)
	// 	// newRows, newCols := newFmtImgs.Dims()

	// 	// // Should be 28 x 28
	// 	// fmt.Println("Dimensions of the new matrix are : ", newRows, newCols)

	// 	// Manually create a 28 x 28 nested slice
	// 	for idx := 0; idx < len(currentImageBytes); idx++ {
	// 		floatNum := float32(currentImageBytes[idx]) // can divide by 255 here if reqd by model
	// 		currentImageFloatVals[idx] = []float32{floatNum}
	// 	}

	// 	for i := 0; i+28 <= len(currentImageFloatVals); i += 28 {
	// 		newFmtImg = append(newFmtImg, currentImageFloatVals[i:i+28])
	// 	}

	// 	// fmt.Println("New 28 x 28 nested slice is :  ", newFmtImg)
	// 	// fmt.Println("Length of nested slice: ", len(newFmtImg))

	// 	// append(append(emptyRows, newFmtImg...), emptyRows...)

	// 	// fmt.Println("New 32 x 32 with padding Image is : ", withPaddingImg)
	// 	// fmt.Println("Length of padded img is: ", len(withPaddingImg))

	// 	finalFmtImg := withPaddingImg

	// 	// fmt.Println("New 32 x 32 with padding Image is : ", finalFmtImg)
	// 	allFinalFmtImgs[i] = finalFmtImg

	// 	if err != nil {
	// 		return 0, 0, nil, err
	// 	}
	// 	if m_ != int(m) {
	// 		return 0, 0, nil, os.ErrInvalid
	// 	}

}

// TEST : SHOW Resized Image in a window

// window := gocv.NewWindow("Hello")
// for {
// 	window.IMShow(resizedImg)
// 	if window.WaitKey(1) >= 0 {
// 		break
// 	}
// }

// width := resizedImg.Cols()

// return pixels
