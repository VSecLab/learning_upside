// static/js/script.js

// funzione per aggiornare le opzioni in "feature" in base al dispositivo selezionato
function updateFeatureOptions() {
    var device = document.getElementById("device").value; // Ottieni il device selezionato
    var featureSelect = document.getElementById("feature"); // Ottieni il campo delle feature
    featureSelect.innerHTML = ""; // Pulisce le opzioni esistenti

    // Controlla quale device è selezionato e aggiorna le opzioni per "feature"
    var options = [];
    if (device === "visore" || device === "controllerDx") {
        options = ["Pitch", "Yaw", "Roll", "X", "Y", "Z"]; // Opzioni per "visore"
    } else if (device === "tablet") {
        options = ["gyroscope.x", "gyroscope.y", "gyroscope.z", "accelerometer.x", "accelerometer.y", "accelerometer.z"]; // Opzioni per "controllerDx"
    }
    var option = document.createElement("option");
    option.value = "";
    option.textContent = "Select a feature";
    featureSelect.appendChild(option);
    
    // aggiungi le nuove opzioni al select "feature"
    for (var i = 0; i < options.length; i++) {
        var option = document.createElement("option");
        option.value = options[i];
        option.textContent = options[i];
        featureSelect.appendChild(option);
    }
}


function setPercentageTo100() {
    var percentageInput = document.getElementById('percentage');
    percentageInput.value = 100;
}


// esegui la funzione quando la pagina viene caricata per la prima volta
window.onload = function() {
    updateFeatureOptions();
}
