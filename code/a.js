function clickAndType() {
    var count = 0;
    var commentBoxes = document.querySelectorAll('div[contenteditable="true"][aria-label^="Bình luận dưới tên"]');
    
    function processCommentBox(index) {
        if (index >= commentBoxes.length) {
            return count;
        }

        // Click vào comment box
        commentBoxes[index].dispatchEvent(new MouseEvent('click', {
            bubbles: true,
            cancelable: true
        }));

        // Input text sau 100ms
        setTimeout(function() {
            var inputEvent = new InputEvent('input', {
                bubbles: true,
                cancelable: true,
                data: 'Up'
            });
            commentBoxes[index].dispatchEvent(inputEvent);
            commentBoxes[index].innerHTML = 'Up';

            // Enter sau 500ms
            setTimeout(function() {
                var enterEvent = new KeyboardEvent('keydown', {
                    key: 'Enter',
                    code: 'Enter',
                    keyCode: 13,
                    which: 13,
                    bubbles: true,
                    cancelable: true
                });
                commentBoxes[index].dispatchEvent(enterEvent);
                count++;

                // Xử lý comment box tiếp theo sau 1000ms
                setTimeout(function() {
                    processCommentBox(index + 1);
                }, 1000);
            }, 500);
        }, 100);
    }

    processCommentBox(0);
    return "Started processing " + commentBoxes.length + " comments";
}

clickAndType();


