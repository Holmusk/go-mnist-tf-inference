package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"path/filepath"

	mnistdata "github.com/petar/GoMNIST"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {

	train, _, err := mnistdata.Load("/Users/kahlil/projects/holmusk/Holmusk Macbook work/Food Scoring Eval/go/GoMNIST/data")
	if err != nil {
		log.Fatal(err)
	}

	// Load a frozen graph to use for queries
	modelpath := filepath.Join("mnistmodel/", "SavedModel.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// model, err := tf.LoadSavedModel("/Users/kahlil/Documents/AIA AI Food Scoring /Data Science team Docs/data_and_model_mnist/model", []string{"serve"}, nil)

	// if err != nil {
	// 	fmt.Printf("Error loading saved model: %s\n", err.Error())
	// 	return
	// }

	// defer model.Session.Close()

	_, terr := dummyInputTensor(28 * 28) // replace this with your own data
	if terr != nil {
		fmt.Printf("Error creating input tensor: %s\n", terr.Error())
		return
	}

	//constTensor
	constTensor, ctErr := constNumTensor(1.0)
	if ctErr != nil {
		fmt.Printf("Error creating const num tensor: %s\n", ctErr.Error())
		return
	}

	sweeper := train.Sweep()
	_, label, _ := sweeper.Next()
	_, label2, _ := sweeper.Next()
	image3, label3, _ := sweeper.Next()

	imageInSlice := [][][][]float32{image3}

	fmt.Println("Image slice passed to tf model: ", imageInSlice)
	imgTensor, imgErr := tf.NewTensor(imageInSlice)
	fmt.Println("Tensor value of the image is: ", imgTensor)
	if imgErr != nil {
		fmt.Printf("Error creating image tensor: %s\n", imgErr.Error())
		return
	}
	// label := 5
	// normalizedImg := make([]float32, 784)
	// for idx, element := range image {
	// 	normalizedImg[idx] = float32(element) / 255
	// }
	// newTensor, newErr := tf.NewTensor(normalizedImg)
	fmt.Println("Label values are: ", label, label2, label3)
	// if newErr != nil {
	// 	fmt.Printf("Error creating Tensor of Image, err: %s\n", newErr.Error())
	// 	return
	// }

	// fakeInput, _ := tf.NewTensor([1][32][32][1]float32{})

	result, runErr := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("IO/input").Output(0):            imgTensor,
			graph.Operation("IO/keep_probability").Output(0): constTensor,
		},
		[]tf.Output{
			graph.Operation("metric/yhat").Output(0),
		},
		nil,
	)

	if runErr != nil {
		fmt.Printf("Error running the session with input, err: %s\n", runErr.Error())
		return
	}

	fmt.Println("Size of the result is: ", len(result))

	fmt.Printf("Most likely number (first prediction from input) is %v \n", result[0].Value())

	// fmt.Println("Done running!", image)

	// fmt.Println("")

	// newMatrix := mat.NewDense(4, 4, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3, 4, 5, 6, 7})

	// fmt.Println(newMatrix)

}

func dummyInputTensor(size int) (*tf.Tensor, error) {

	// imageData := [][]float32{make([]float32, size)}
	newImagedata := [1][32][32][1]float32{}
	fmt.Println("Fake Image slice : ", newImagedata)
	return tf.NewTensor(newImagedata)
}

func constNumTensor(numVal float32) (*tf.Tensor, error) {
	return tf.NewTensor(numVal)
}

// pred_value = sess.run(['prefix/metric/yhat:0'], feed_dict={'prefix/IO/input:0': X_test, 'prefix/IO/keep_probability:0':1.0})

// result, runErr := model.Session.Run(
// 	map[tf.Output]*tf.Tensor{
// 		model.Graph.Operation("imageinput").Output(0): tensor,
// 	},
// 	[]tf.Output{
// 		model.Graph.Operation("infer").Output(0),
// 	},
// 	nil,
// )
