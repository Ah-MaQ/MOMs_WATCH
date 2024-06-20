document.addEventListener("DOMContentLoaded", () => {
    const API_URL = "https://secret-journey-41438-8aa9540f2edc.herokuapp.com";
    fetch(API_URL + "/read/test2")
    .then((response) => response.json())
    .then((data) => {
        const dataContainer = document.getElementById("database-container");
        dataContainer.innerText = JSON.stringify(data, null, 2);
    })
    .catch((error) => {
        console.error('Error fetching data:', error);
    });
});
