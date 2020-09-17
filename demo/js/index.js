// welcome
var page1 = document.getElementById("page1");
// display
var page2 = document.getElementById("page2");
// directory of the result figures
var result_dir = "";
// index of the current figure
var img_index = -1;
// files
init_files = [];
target_files = [];
blocks_num = 10;


page1.style.display = "block";
document.getElementById('restartBtn').style.display = 'none';
document.getElementById('prevBtn').style.display = 'none';

function next() {
    if (page1.style.display === "block") {
        page1.style.display = "none";
        page2.style.display = "block";
        document.getElementById('init_img').src = '/demo/loading.png';
        document.getElementById('target_img').src = '/demo/loading.png';
        document.getElementById('nextBtn').style.display = 'none';
        send_start();
    }
    else if (page2.style.display = "block") {
        document.getElementById("nextBtn").innerHTML = "Next";
        setNewImage(1);
    }
}

function prev() {
    if (page2.style.display = "block") {
        setNewImage(-1);
    }
}

function restart() {
    page2.style.display = 'none';
    page1.style.display = 'block';
    document.getElementById('restartBtn').style.display = 'none';
    document.getElementById('prevBtn').style.display = 'none';
    document.getElementById('nextBtn').style.display = 'block';
    document.getElementById("nextBtn").innerHTML = "Start";

    result_dir = "";
    img_index = -1;
    init_files = [];
    target_files = [];
}

function send_start() {
    var now = new Date();
    var year = now.getFullYear();
    var month = now.getMonth();
    var date = now.getDate();
    var hour = now.getHours();
    var minute = now.getMinutes();
    var second = now.getSeconds();
    result_dir = 'test_result/' + String(year) + '-' + String(month) + '-' + String(date) + '-' + String(hour) + '-' + String(minute) + '-' + String(second);

    $.ajax({
        url: '/startsign',
        type: 'POST',
        data: JSON.stringify(
            {
                'save_dir': result_dir
            }
        ),
        success: function (res) {
            page2.style.display = "block";
            document.getElementById("nextBtn").innerHTML = "Next";
            document.getElementById('nextBtn').style.display = 'block';

            for (var i = 0; i < blocks_num+1; i++) {
                init_files.push(result_dir + '/init_' + String(i) + '.png');
                target_files.push(result_dir + '/target_' + String(i) + '.png');
            }
            setNewImage(1);

            console.log(res);
            console.log(0);
        },
        error: function (res) {
            console.log(res);
            console.log(1);
        }
    })
}

function setNewImage(count) {

    img_index = img_index + count;

    if (img_index === blocks_num) {
        document.getElementById('nextBtn').style.display = 'none';
        document.getElementById('prevBtn').style.display = 'block';
        document.getElementById('restartBtn').style.display = 'block';
    }
    else if (img_index === 0){
        document.getElementById('nextBtn').style.display = 'block';
        document.getElementById('prevBtn').style.display = 'none';
    }
    else {
        document.getElementById('nextBtn').style.display = 'block';
        document.getElementById('prevBtn').style.display = 'block';
    }
    var init = init_files[img_index];
    var target = target_files[img_index];

    try {
        
        document.getElementById('init_img').style.height = '300px';
        document.getElementById('target_img').style.height = '300px';

        // document.getElementById('init_img').src = '/demo/loading.png'
        // document.getElementById('target_img').src = '/demo/loading.png'

        document.getElementById('init_img').src = init;
        document.getElementById('target_img').src = target;

    } catch (error) {
        console.log('show image error');
    }

}