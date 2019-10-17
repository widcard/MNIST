import io.DataSetUtil;
import io.ModelUtil;
import model.Data;
import model.MatrixGenerator;
import nn.*;


import java.util.*;
/**
 * Name:Chaoqun Yu
 * ID:10242826	
 * Data:10/16/2019
 * Assignment 2
 * Description:recognizes the MNNIST digit set using 3 layer neural network
 *
 */
public class App {

    private static final int WIDTH = 28;
    private static final int HEIGHT = 28;
    private static final int INPUT_SIZE = WIDTH * HEIGHT;
    private static final int LAYER1_SIZE = 32;
    private static final int LAYER2_SIZE = 16;
    private static final int OUTPUT_SIZE = 10;

    private static final int BATCH_SIZE = 10;
    private static final int EPOCH = 30;

    private static final double LEARNING_RATE = 3.0;

    private static final String MODEL_NAME = "mnist.csv";
    private static final String TRAIN_SET_NAME = "mnist_train.csv";
    private static final String TEST_SET_NAME = "mnist_test.csv";

    /**
     * print menu
     */
    private static void printMenu() {
        System.out.println("Please enter a choice[0-5]");
        System.out.println("===================================================");
        System.out.println("\t1. train a network");
        System.out.println("\t2. load a network from local file");
        System.out.println("\t3. show the network's accuracy on train-set");
        System.out.println("\t4. show the network's accuracy on test-set");
        System.out.println("\t5. save the network to local file");
        System.out.println("\t0. exit");
        System.out.println("===================================================");
        System.out.print("Enter your choice: ");
    }

    private static List<Data> mListCorrect = new ArrayList<>();
    private static Map<Data, Integer> mMapInCorrect = new HashMap<>();

    /**
     * print the accuracy
     *
     * @param neuralNet network
     * @param dataSet   data set
     */
    private static void accuracy(NeuralNet neuralNet, List<Data> dataSet) {
        int[] sum = new int[OUTPUT_SIZE];
        int[] right = new int[OUTPUT_SIZE];

        int total = 0;
        int totalRight = 0;
        for (Data data : dataSet) {
            sum[data.getLabel()]++;
            total++;
            int result = neuralNet.getLabel(data);
            if (result == data.getLabel()) {
                right[result]++;
                totalRight++;
                mListCorrect.add(data);
            } else {
                mMapInCorrect.put(data, result);
            }
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            System.out.print(i + " = " + right[i] + "/" + sum[i] + "\t");
        }
        System.out.println();
        System.out.println("Accuracy = " + totalRight + "/" + total + " = " + String.format("%.3f%%", totalRight * 1.0 / total * 100));
    }

    /**
     * get the data-set-util and print
     *
     * @param fileName file name
     * @return util
     */
    private static DataSetUtil getDataSetUtil(String fileName) {

        return new DataSetUtil(fileName);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        DataSetUtil fTrain = null;
        DataSetUtil fTest = null;
        NeuralNet nn = null;
        while (true) {
            printMenu();
            String s = scanner.next();
            switch (Integer.parseInt(s)) {
                case 1:
                    if (fTrain == null) {
                        fTrain = new DataSetUtil(TRAIN_SET_NAME);
                    }
                    nn = new NeuralNet(LEARNING_RATE, BATCH_SIZE);

                    // layer1 hidden layer
                    Layer layer1 = new Layer();
                    layer1.setWeights(MatrixGenerator.randn(LAYER1_SIZE, INPUT_SIZE));
                    layer1.setBias(MatrixGenerator.randn(LAYER1_SIZE, 1));

                    // layer2 hidden layer
                    Layer layer2 = new Layer();
                    layer2.setWeights(MatrixGenerator.randn(LAYER2_SIZE, LAYER1_SIZE));
                    layer2.setBias(MatrixGenerator.randn(LAYER2_SIZE, 1));
              

                    Layer layer3 = new Layer();
                    layer3.setWeights(MatrixGenerator.randn(OUTPUT_SIZE, LAYER2_SIZE));
                    layer3.setBias(MatrixGenerator.randn(OUTPUT_SIZE, 1));

                    nn.addLayers(layer1, layer2, layer3);

                    for (int i = 0; i < EPOCH; i++) {
                        int[][] result = nn.train(fTrain.loadData());
                        System.out.println("Epoch: " + (i + 1));
                        int total = 0;
                        int totalRight = 0;
                        for (int j = 0; j < OUTPUT_SIZE; j++) {
                            total += result[0][j];
                            totalRight += result[1][j];
                            System.out.print(j + " = " + result[1][j] + "/" + result[0][j] + "\t");
                    
                        }
                        System.out.println();
                        System.out.println(String.format("Accuracy = %d/%d = %.3f%%", totalRight, total, (totalRight * 1.0 / total) * 100));
                        
                    }
                    break;
                case 2:
                    nn = ModelUtil.loadNetWork(MODEL_NAME);
                    if (nn == null) {
                        System.out.println("load model failed");
                    } else {
                        System.out.println("load model successfully");
                    }
                    break;
                case 3:
                    if (nn != null) {
                        if (fTrain == null) {
                            fTrain = getDataSetUtil(TRAIN_SET_NAME);
                        }
                        accuracy(nn, fTrain.loadData());
                    } else {
                        System.out.println("no trained network");
                    }

                    break;
                case 4:
                    if (nn != null) {
                        if (fTest == null) {
                            fTest = getDataSetUtil(TEST_SET_NAME);
                        }
                        accuracy(nn, fTest.loadData());
                    } else {
                        System.out.println("no trained network");
                    }
                    break;
               
                case 5:
                    if (nn != null) {
                        boolean b = ModelUtil.saveNetWork(nn, MODEL_NAME);
                        if (!b) {
                            System.out.println("save model failed");
                        } else {
                            System.out.println("save model successfully");
                        }
                    } else {
                        System.out.println("no trained network");
                    }
                    break;
                default:
                    return;
            }
            System.out.println();
        }
    }


}
