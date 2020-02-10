package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"path/filepath"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func runInferenceModel(imgPaths []string, imgsForTensor [][3][224][224]float32) {

	// Load a frozen graph to use for queries
	// modelpath := filepath.Join("mnistmodel/", "SavedModel.pb")
	// trial_modelPath := filepath.Join("/Users/kahlil/projects/holmusk/Food Scoring Eval/go/mnist_infer/models/trial_model/", "model.pb")
	food_notFoodModelPath := filepath.Join("/Users/kahlil/Documents/AIA AI Food Scoring /S3 AIA models/Phase1/TF/models/FoodOrNonfood/", "epoch_149.pb")
	model, err := ioutil.ReadFile(food_notFoodModelPath)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}
	// graphInputOp := graph.Operation("IO/input")
	// fmt.Printf("List of operations in the graph: ", graphInputOp)

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// //constTensor (for probability)
	// _, ctErr := constNumTensor(1.0)
	// if ctErr != nil {
	// 	fmt.Printf("Error creating const num tensor: %s\n", terr.Error())
	// 	return
	// }

	newTensor, newErr := tf.NewTensor(imgsForTensor)
	if newErr != nil {
		fmt.Printf("Error creating Tensor of Image, err: %s\n", newErr.Error())
		return
	}
	result, runErr := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("IO/input").Output(0): newTensor,
		},
		[]tf.Output{
			graph.Operation("IO/output").Output(0),
		},
		nil,
	)

	if runErr != nil {
		fmt.Printf("Error running the session with input, err: %s\n", runErr.Error())
		return
	}

	// var tensorValInGo [2]float32
	tensorValInGo := result[0].Value()
	logitsVal, _ := tensorValInGo.([][]float32)
	for idx := 0; idx < len(logitsVal); idx++ {
		logitSlice := logitsVal[idx]
		firstLogit, secondLogit := logitSlice[0], logitSlice[1]
		var resultOfModel int
		if firstLogit > secondLogit {
			resultOfModel = 0
		} else if firstLogit < secondLogit {
			resultOfModel = 1
		} else {
			resultOfModel = -1
		}
		fmt.Printf("Img: %v  ", imgPaths[idx])
		fmt.Printf("Logits:  %v %v Result: %v \n", firstLogit, secondLogit, resultOfModel)
	}

}

func dummyInputTensor(size int) (*tf.Tensor, error) {

	// imageData := [][]float32{make([]float32, size)}
	newImagedata := [1][3][224][224]float32{}
	return tf.NewTensor(newImagedata)
}

// func constNumTensor(numVal float32) (*tf.Tensor, error) {
// 	return tf.NewTensor(numVal)
// }

// pred_value = sess.run(['prefix/metric/yhat:0'], feed_dict={'prefix/IO/input:0': X_test, 'prefix/IO/keep_probability:0':1.0})
