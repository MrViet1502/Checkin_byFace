// Đăng ký người dùng
document.getElementById('registerForm').addEventListener('submit', function (e) {
    e.preventDefault();
    const formData = new FormData(e.target);

    fetch('/register', {
        method: 'POST',
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            document.getElementById('registerResult').innerText = data.message;
            window.alert(data.message);
        })
        .catch((error) => console.error('Error:', error));
});

// Check-in bằng camera
document.getElementById('checkInBtn').addEventListener('click', () => {
    fetch('/check_in', {
        method: 'GET',
    })
        .then((response) => response.json())
        .then((data) => {
            const resultDiv = document.getElementById('checkInResult');
            resultDiv.innerHTML = `<p>${data.message}</p>`;
            window.alert(data.message);
        })
        .catch((error) => console.error('Error:', error));
});

// Lấy danh sách check-in
document.getElementById('fetchListBtn').addEventListener('click', () => {
    fetch('/check_in_list', {
        method: 'GET',
    })
        .then((response) => response.json())
        .then((data) => {
            const resultDiv = document.getElementById('checkInListResult');
            resultDiv.innerHTML = '<h3>Check-In List:</h3>';
            data.data.forEach((entry) => {
                resultDiv.innerHTML += `<p>Name: ${entry.name}, Time: ${entry.time}</p>`;
            });
        })
        .catch((error) => console.error('Error:', error));
});
