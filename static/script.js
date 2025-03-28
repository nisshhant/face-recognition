document.addEventListener("DOMContentLoaded", function () {
    const cameraContainer = document.getElementById("camera-container");
    const videoFeed = document.getElementById("videoFeed");
    const startCamera = document.getElementById("startCamera");
    const stopCamera = document.getElementById("stopCamera");

    startCamera.addEventListener("click", function () {
        cameraContainer.style.display = "flex"; // Show camera feed
        
    });

    stopCamera.addEventListener("click", function () {
        cameraContainer.style.display = "none";
    });





    const enrollSection = document.getElementById("enrollSection");
    

    // Handle enrollment form submission
    enrollForm.addEventListener("submit", function (event) {
        event.preventDefault();

        let formData = new FormData();
        formData.append("name", document.getElementById("name").value);
        formData.append("user_id", document.getElementById("user_id").value);

        fetch("/enroll", {
            method: "POST",
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                messageBox.textContent = data.message;
                messageBox.style.color = data.message.includes("saved") ? "green" : "red";
            })
            .catch(error => {
                console.error("Error:", error);
                messageBox.textContent = "Enrollment failed!";
                messageBox.style.color = "red";
            });
    });



    startCamera.addEventListener("click", function (event) {

    });
});
