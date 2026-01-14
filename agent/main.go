package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

var SENZOR_NUM = 228
var SENZOR_DATA_BATCH = 12
var RETRAIN_DATA_NUM = 24
var (
	SENZOR_DATA_URL string
	PREDICT_URL     string
	ADD_DATA_URL    string
	RETRAIN_URL     string
	HEALTH_URL      string
)

func loadEnv() error {
	file, err := os.Open(".env")
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			log.Println("Skipping invalid line:", line)
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		value = strings.Trim(value, `"'`)

		os.Setenv(key, value)
	}

	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

func init() {
	if os.Getenv("LOAD_ENV_FILE") == "" {
		if err := loadEnv(); err != nil {
			log.Fatalln("Could not load env.", err)
		}
	}

	url := os.Getenv("API_URL")
	PREDICT_URL = fmt.Sprintf("%s/predict", url)
	ADD_DATA_URL = fmt.Sprintf("%s/add-data", url)
	RETRAIN_URL = fmt.Sprintf("%s/train", url)
	HEALTH_URL = fmt.Sprintf("%s/health", url)

	url = os.Getenv("SENSOR_DATA_URL")
	SENZOR_DATA_URL = fmt.Sprintf("%s/sensor-data", url)
}

type Prediction struct {
	Preds [][]float64 `json:"predictions"`
}

type HealthResponse struct {
	Status string `json:"status"`
	Device string `json:"device"`
}

func waitApi() {
	fmt.Println("Waiting for model to be ready...")
	for {
		time.Sleep(5 * time.Second)

		resp, err := http.Get(HEALTH_URL)
		if err != nil {
			continue
		}

		var data HealthResponse
		if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
			resp.Body.Close()
			continue
		}
		resp.Body.Close()

		if data.Status == "Ready" {
			fmt.Println("Model is Ready")
			break
		}
	}
}

func getSenzorReading() ([][]float64, error) {
	result := make([][]float64, 0, SENZOR_DATA_BATCH)

	for i := 0; i < SENZOR_DATA_BATCH; i++ {
		resp, err := http.Get(SENZOR_DATA_URL)
		if err != nil {
			return nil, err
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}

		strReadings := strings.Split(strings.TrimSpace(string(body)), ",")

		if len(strReadings) != SENZOR_NUM {
			return nil, errors.New("invalid sensor count")
		}

		readings := make([]float64, SENZOR_NUM)
		for j, s := range strReadings {
			val, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
			if err != nil {
				return nil, err
			}
			readings[j] = val
		}

		result = append(result, readings)
	}

	return result, nil
}

func getPredictions(r [][]float64) ([][]float64, error) {
	body := map[string][][]float64{
		"data": r,
	}

	jsonBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := http.Post(PREDICT_URL, "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("failed to add data")
	}

	var data Prediction
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}

	return data.Preds, nil
}

func addData(r [][]float64) error {
	body := map[string][][]float64{
		"measurements": r,
	}

	jsonBytes, err := json.Marshal(body)
	if err != nil {
		return err
	}

	resp, err := http.Post(ADD_DATA_URL, "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return errors.New("failed to add data")
	}

	return nil
}

func retrain() error {
	resp, err := http.Post(RETRAIN_URL, "application/json", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return errors.New("failed to retrain")
	}

	return nil
}

func main() {
	waitApi()

	var readingsBatch [][]float64

	prevReadings, err := getSenzorReading()
	if err != nil {
		log.Fatalln("Could not get readings:", err)
	}
	fmt.Println("Got senzors readings...")
	for _, el := range prevReadings {
		readingsBatch = append(readingsBatch, el)
	}

	preds, err := getPredictions(prevReadings)
	if err != nil {
		log.Fatalln("Could not get predictions:", err)
	}
	fmt.Println("\nGot predictions...")

	for {
		r, err := getSenzorReading()
		if err != nil {
			log.Fatalln("Could not get readings:", err)
		}
		fmt.Println("\nGot senzors readings...")

		fmt.Print("\nSensor 1 predictions: ")
		for i, p := range preds {
			fmt.Printf("%.2f", p[0])
			if i != len(preds)-1 {
				fmt.Print(", ")
			} else {
				fmt.Print("\n")
			}
		}

		fmt.Print("Sensor 1 readings:    ")
		for i, r := range prevReadings {
			fmt.Printf("%.2f", r[0])
			if i != len(prevReadings)-1 {
				fmt.Print(", ")
			} else {
				fmt.Print("\n")
			}
		}

		for _, el := range r {
			readingsBatch = append(readingsBatch, el)
		}
		prevReadings = r

		preds, err = getPredictions(prevReadings)
		if err != nil {
			log.Fatalln("Could not get predictions:", err)
		}
		fmt.Println("\nGot predictions...")

		if len(readingsBatch) >= RETRAIN_DATA_NUM {
			if err = addData(readingsBatch); err != nil {
				log.Fatalln("Could not add data:", err)
			}
			fmt.Println("\nData added")

			if err = retrain(); err != nil {
				log.Fatalln("Could not retrain:", err)
			}
			fmt.Println("\nModel update scheduled. Sleeping...")
			time.Sleep(40 * time.Second)

			readingsBatch = [][]float64{}
		}
	}
}
