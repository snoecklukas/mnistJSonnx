const argMax = function(array) {
    return [].reduce.call(array, (m, c, i, arr) => c > arr[m] ? i : m, 0);
};

const readURL = function() {
    if (this.files && this.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            document.querySelector("#image_preview").src = e.target.result;
        }

        reader.readAsDataURL(this.files[0]); // base64 string
    }
};

const showResults = function(onnxOutput) {
    let domPrediction = document.querySelector("#prediction");
    let domScores = document.querySelector("#scores");

    console.log(onnxOutput.data);
    let scoreString = "";
    onnxOutput.data.forEach(function(value, index) {
        scoreString += `${index}: ${value}\n`;
    });

    domPrediction.innerHTML = argMax(onnxOutput.data);
    domScores.innerHTML = scoreString;
};

const clearData = function() {
    document.querySelector("#image_upload").value = '';
    document.querySelector("#image_preview").src = "An image";
    document.querySelector("#prediction").text = '?';
    document.querySelector("#scores").value = '';
};

const checkData = function() {
    console.log("check");
    let domImageUpload = document.querySelector("#image_upload");
    let text = "";
    if('files' in domImageUpload) {
        if(domImageUpload.files.length == 0) {
            text = "Select one or more files";
        } else {
            let reader = new FileReader();
            if(domImageUpload.files[0]) {
                reader.onload = () => {
                    imgBase64 = reader.result;
                    console.log(reader.result);
                    console.warn(imgBase64);

                    getInfo(reader.result)

                    //consumeONNXSession();
                }
                console.log(domImageUpload.files[0])
                reader.readAsDataURL(domImageUpload.files[0]);
            }
        }
    }
};

const getInfo = function(pic) {
	var ctxt = canvas.getContext('2d');
	var img = new Image;
	img.src = pic;
	img.onload = function() {
		ctxt.drawImage(img, 0, 0);
        var data = ctxt.getImageData(0, 0, img.width, img.height).data;
        counter = 0;
        rgb_array = [];
        new_data = [];
        for(pixel of data) {
            counter ++;
            if(counter % 4 == 0) {
                new_pixel = (0.3 * rgb_array[0] + 0.59 * rgb_array[1] + 0.11 * rgb_array[2]) / 255;
                new_data.push(new_pixel);
                rgb_array = [];
                counter = 0;
            } else {
                rgb_array[counter-1] = pixel;
            }
        };
        console.log(new_data);
        consumeONNXSession(new_data);
	}
}

const listenToCheck = function() {
    let domCheckButton = document.querySelector("#check_upload");
    domCheckButton.addEventListener('click', checkData);
};

const listenToClear = function() {
    let domClearButton = document.querySelector("#clearArea_upload");
    domClearButton.addEventListener('click', clearData);
};

const listenToUpload = function() {
    let domImageUpload = document.querySelector("#image_upload");
    domImageUpload.addEventListener('change', readURL);
};

const consumeONNXSession = async function(pixelArray) {
    // create session
    const myOnnxSession = new onnx.InferenceSession();
    // load the ONNX model file
    await myOnnxSession.loadModel("./asset/mnist-onnx-js-v1.onnx");
    // process the array
    // let newArr = [];
    // while(pixelArray.length) newArr.push(pixelArray.splice(0,28));
    const inputs = [
        new Tensor(new Float32Array(pixelArray), "float32", [1, 28, 28, 1]),
      ];
    //console.log(inputs);
    // predict
    const outputMap = await myOnnxSession.run(inputs);
    const outputData = outputMap.get('mnist_output');
    console.log(outputData);
    
    showResults(outputData);
};

const init = function () {
    console.log('DOM loaded');

    document.querySelector("#image_upload").value = '';
    document.querySelector("#image_preview").src = "An image";

    listenToUpload();
    listenToClear();
    listenToCheck();
};

document.addEventListener('DOMContentLoaded', init);