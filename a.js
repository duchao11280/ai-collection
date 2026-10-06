// var iframe = document.querySelectorAll('iframe');
// var url;
// for (let i = 0; i < iframe.length; i++) {
//     if (iframe[i].hasAttribute('src')){
//         if (iframe[i].src.includes('tgWebAppData')){
//             url = iframe[i].src;
//             break;
//         }
//     }
// }

// const hash = url.split('#')[1];

// const params = new URLSearchParams(hash);
// const tgWebAppData = params.get('tgWebAppData');

// return tgWebAppData


function clickAt(x, y) {
    const element = document.elementFromPoint(x, y);
    if (element) {
        const clickEvent = new MouseEvent('click', {
        view: window,
        bubbles: true,
        cancelable: true,
        clientX: x,
        clientY: y
        });
        element.dispatchEvent(clickEvent);
    } else {
        console.log('No element found at the specified coordinates');
    }
}