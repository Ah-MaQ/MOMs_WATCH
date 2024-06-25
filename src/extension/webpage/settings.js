<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Settings Page</title>
    <link rel="stylesheet" href="styles.css">
    <script src="settings.js" defer></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('nav ul li a').forEach(function(link) {
                link.addEventListener('click', function(event) {
                    event.preventDefault();
                    window.open(link.href, '_blank', 'width=800,height=600');
                });
            });
        });
    </script>
</head>
<body>
    <nav>
        <ul>
            <li><a href="main.html">메인</a></li>
            <li><a href="focus.html">집중기록</a></li>
            <li><a href="settings.html">설정</a></li>
        </ul>
    </nav>
    <main>
        <h1>설정 페이지</h1>
        <p>여기는 설정 페이지 내용입니다.</p>
    </main>
</body>
</html>
