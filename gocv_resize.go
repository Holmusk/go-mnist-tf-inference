package main

import (
	"fmt"
	"image"
	"log"

	"gocv.io/x/gocv"
)

func main() {

	// notFoodImgPath := "/Users/kahlil/Pictures/plate-food.jpg"
	foodImgPath := "/Users/kahlil/Documents/AIA AI Food Scoring /Food NotFood testing/Test1 /Images for testing/8453869e-d8ce-4009-8011-8a20b58b2037-large-thumb.jpg"

	filename := foodImgPath

	img := gocv.IMRead(filename, gocv.IMReadColor)
	if img.Empty() {
		fmt.Printf("Error reading image from: %v\n", filename)
		return
	}

	resizedImg := gocv.NewMat()

	gocv.Resize(img, &resizedImg, image.Pt(224, 224), 0.0, 0.0, gocv.InterpolationCubic)

	imgPixels, errMsg := reshapeImgPixelsForTensor(resizedImg)
	if errMsg != nil {
		log.Fatal(errMsg)
	}
	var singleImgPixelArray [1][3][224][224]float32
	singleImgPixelArray[0] = imgPixels
	runInferenceModel(singleImgPixelArray)
	fmt.Println()
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
			B := float32(uint8(imgData[y*step+x])) / 255.0
			G := float32(uint8(imgData[y*step+x+1])) / 255.0
			R := float32(uint8(imgData[y*step+x+2])) / 255.0
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
	allChannelsImgPixels[0] = bluePixels
	allChannelsImgPixels[1] = greenPixels
	allChannelsImgPixels[2] = redPixels

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
