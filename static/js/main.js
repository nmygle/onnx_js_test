// math functions
function argsort(array) {
    const arrayObject = array.map((value, idx) => { return { value, idx }; });
    arrayObject.sort((a, b) => {
        if (a.value < b.value) {
            return -1;
        }
        if (a.value > b.value) {
            return 1;
        }
        return 0;
    });
    const argIndices = arrayObject.map(data => data.idx);
    return argIndices;
}

function softmax(arr) {
    const C = Math.max(...arr);
    const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
    return arr.map((value, index) => { 
        return Math.exp(value - C) / d;
    })
}

// load onnx data
const session = new onnx.InferenceSession({backendHint: "cpu"});
session.loadModel("/static/mobilenetv2-1.0.onnx");
let width = 224;
let height = 224;

// vue functions
new Vue({
    el: "#app",
    data: {
        uploadedImage: "",
        synset: "",
        results: [[0,0]],
        start_calc: false,
        end_calc: false,
    },
    delimiters: ["<{", "}>"],
    mounted() {
        // load category data
        axios.get("/static/synset.json").then(response => (this.synset = response.data));
    },
    methods: {
        onFileChange(e){
            let files = e.target.files || e.dataTransfer.files;
            this.createImage(files[0]);
        },
        createImage(file){
            console.log("start calc");
            this.start_calc = true;
            this.end_calc = false;
            let reader = new FileReader();
            reader.onload = (e) => {
                this.uploadedImage = e.target.result;
                let canvas = document.createElement("canvas");
                let context = canvas.getContext("2d");
                let img = new Image;
                img.src = this.uploadedImage;
                img.onload = (e) => {
                    context.drawImage(img, 0, 0, img.width, img.height);
                    let img_min = Math.min(img.width, img.height);
                    if  (img_min != width){
                        rate = width / img_min;
                        dwidth = parseInt(img.width * rate);
                        dheight = parseInt(img.height * rate);
                        context.drawImage(img, 0, 0, img.width, img.height, 0, 0, dwidth, dheight);
                    } else {
                        dwidth = img.width;
                        dheight = img.height;
                    }
                    if (dwidth > dheight){
                        dx = parseInt((dwidth - width) / 2);
                        img = context.getImageData(dx, 0, width, width);
                    } else {
                        dy = parseInt((dheight - width) / 2);
                        img = context.getImageData(0, dy, width, width);
                    }

                    // image to tensor
                    let imgData = new Float32Array(img.data);
                    let input = new Float32Array(width * height * 3).fill(1.0);
                    let count = 0
                    for (i=0; i<imgData.length; i+=4){
                        input[count] = (imgData[i]/255 - 0.485)/0.229;
                        count += 1;
                    }
                    for (i=0; i<imgData.length; i+=4){
                        input[count] = (imgData[i+1]/255 - 0.456)/0.224;
                        count += 1;
                    }
                    for (i=0; i<imgData.length; i+=4){
                        input[count] = (imgData[i+2]/255 - 0.406)/0.225;
                        count += 1;
                    }
                    input = [new Tensor(input, "float32", [1, 3, width, height])];
                    
                    // predict
                    session.run(input).then(output => {
                        const outputTensor = output.values().next().value;
                        const proba = softmax(Array.prototype.slice.call(outputTensor.data));
                        const rank = argsort(proba).reverse().slice(0, 5);
                        let results = [];
                        for(i=0; i<5; i++){
                            label0 = this.synset[rank[i]];
                            proba0 = proba[rank[i]];
                            results.push([label0, proba0]);
                        }
                        this.results = results;
                        this.start_calc = false;
                        this.end_calc = true;
                        console.log("finish");
                    });
                }
            };
            reader.readAsDataURL(file);
        }
    },
    filters: {
        probaDecimal(value){
            return (value*100).toFixed(2);
        }
    }
});
