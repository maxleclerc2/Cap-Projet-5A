// File Upload
// 
function ekUpload(){
  function Init() {

    console.log("Upload Initialised");

    var fileSelect    = document.getElementById('file-upload'),
        fileDrag      = document.getElementById('file-drag');
        //submitButton  = document.getElementById('submit-button');

    fileSelect.addEventListener('change', fileSelectHandler, false);

    // Is XHR2 available?
    var xhr = new XMLHttpRequest();
    if (xhr.upload) {
      // File Drop
      fileDrag.addEventListener('dragover', fileDragHover, false);
      fileDrag.addEventListener('dragleave', fileDragHover, false);
      fileDrag.addEventListener('drop', fileSelectHandler, false);
    }
  }

  function fileDragHover(e) {
    var fileDrag = document.getElementById('file-drag');

    e.stopPropagation();
    e.preventDefault();

    fileDrag.className = (e.type === 'dragover' ? 'hover' : 'modal-body file-upload');
  }

  function fileSelectHandler(e) {
    // Fetch FileList object
    var files = e.target.files || e.dataTransfer.files;

    // Cancel event and hover styling
    fileDragHover(e);

    // Process all File objects
    for (var i = 0, f; f = files[i]; i++) {
      parseFile(f);
      uploadFile(f);
    }
  }

  // Output
  function output(msg) {
    // Response
    var m = document.getElementById('messages');
    m.innerHTML = msg;
  }

  function parseFile(file) {

    console.log(file.name);
    output(
      '<strong>' + encodeURI(file.name) + '</strong>'
    );
    
    // var fileType = file.type;
    // console.log(fileType);
    var imageName = file.name;

    var isGood = (/\.(?=txt|csv|xlsx|png|jpg|jpeg|tiff)/gi).test(imageName);
    if (isGood) {
      document.getElementById('start').classList.add("hidden");
      document.getElementById('response').classList.remove("hidden");
      document.getElementById('notimage').classList.add("hidden");
      // Thumbnail Preview
      document.getElementById('file-upload').classList.remove("hidden");
      document.getElementById('file-upload').src = URL.createObjectURL(file);
    }
    else {
      document.getElementById('file-upload').classList.add("hidden");
      document.getElementById('notimage').classList.remove("hidden");
      document.getElementById('start').classList.remove("hidden");
      document.getElementById('response').classList.add("hidden");
      document.getElementById("uploader").reset();
    }
  }

  function setProgressMaxValue(e) {
    var pBar = document.getElementById('file-progress');

    if (e.lengthComputable) {
      pBar.max = e.total;
    }
  }

  function updateFileProgress(e) {
    var pBar = document.getElementById('file-progress');

    if (e.lengthComputable) {
      pBar.value = e.loaded;
    }
  }

  function uploadFile(file) {

    var xhr = new XMLHttpRequest(),
      fileInput = document.getElementById('class-roster-file'),
      pBar = document.getElementById('file-progress'),
      fileSizeLimit = 1024; // In MB
    if (xhr.upload) {
      // Check if file is less than x MB
      if (file.size <= fileSizeLimit * 1024 * 1024) {
        // Progress bar
        pBar.style.display = 'inline';
        xhr.upload.addEventListener('loadstart', setProgressMaxValue, false);
        xhr.upload.addEventListener('progress', updateFileProgress, false);

        // File received / failed
        xhr.onreadystatechange = function(e) {
          if (xhr.readyState == 4) {
            // Everything is good!

            // progress.className = (xhr.status == 200 ? "success" : "failure");
            // document.location.reload(true);
          }
        };

        // Start upload
        xhr.open('POST', document.getElementById('uploader').action, true);
        xhr.setRequestHeader('X-File-Name', file.name);
        xhr.setRequestHeader('X-File-Size', file.size);
        xhr.setRequestHeader('Content-Type', 'multipart/form-data');
        xhr.send(file);
      } else {
        output('Please upload a smaller file (< ' + fileSizeLimit + ' MB).');
      }
    }
  }

  // Check for the various File API support.
  if (window.File && window.FileList && window.FileReader) {
    Init();
  } else {
    document.getElementById('file-drag').style.display = 'none';
  }
}
ekUpload();

function showFields(checkboxID, fiel1ID, field2ID) {
    // Get the checkbox
    var checkBox = document.getElementById(checkboxID);
    // Get the output text
    var field1 = document.getElementById(fiel1ID);
    var field2 = document.getElementById(field2ID);
  
    // If the checkbox is checked, display the output text
    if (checkBox.checked == true){
        field1.style.display = "block";
        field2.style.display = "block";
    } else {
        field1.style.display = "none";
        field2.style.display = "none";
    }
  } 

function showForm(checkboxID, formID) {
    // Get the checkbox
    var checkBox = document.getElementById(checkboxID);
    // Get the output text
    var form = document.getElementById(formID);
  
    // If the checkbox is checked, display the output text
    if (checkBox.checked == true){
        form.style.display = "block";
        checkBox.scrollIntoView({behavior: 'smooth'});
    } else {
        form.style.display = "none";
    }
  } 

function showSigma() {
    var radioID = document.getElementById("Seuil-1")
    var selectSigma = document.getElementById("selectSigma")
    var manualValue = document.getElementById("manualValue")

    if(radioID.checked == true) {
        selectSigma.style.display = "block";
        manualValue.style.display = "none";
    } else {
        selectSigma.style.display = "none";
        manualValue.style.display = "block";
    }
}

function showSigmaBis() {
  var radioID = document.getElementById("SeuilBis-1")
  var selectSigma = document.getElementById("selectSigmaBis")
  var manualValue = document.getElementById("manualValueBis")

  if(radioID.checked == true) {
      selectSigma.style.display = "block";
      manualValue.style.display = "none";
  } else {
      selectSigma.style.display = "none";
      manualValue.style.display = "block";
  }
}


jQuery('#DateBegin').datetimepicker({
  timepicker: false,
  format: 'Y-m-d',
  formatDate: 'Y-m-d',
  maxDate: '0'
});

jQuery('#DateEnd').datetimepicker({
  timepicker: false,
  format: 'Y-m-d',
  formatDate: 'Y-m-d',
  maxDate: '0'
});

function showAddForm(value) {
  if (value == "1") {
    document.getElementById("formAddSat").style.display = "inline";
  } else {
    document.getElementById("formAddSat").style.display = "none";
  }
};