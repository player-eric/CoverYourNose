const base64Prefix = "data:image/png;base64,";
const placeHolderImagePath = "static/logo.png";

const handleImageUpload = (event) => {
	const files = event.target.files;
	const formData = new FormData();
	formData.append("file", files[0]);
	console.log("uploaded an image");
	fetch("/predict", {
		method: "POST",
		body: formData,
	})
		.then((response) => response.json())
		.then((data) => {
			document.getElementById("result-image").src =
				base64Prefix + data.image;
			document.getElementById("upload-label").innerHTML = data.message;
		})
		.catch((error) => {
			console.error(error);
		});
};
