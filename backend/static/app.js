async function searchData() {
    const fromDate = document.querySelector('input[name="fromDate"]').value;
    const toDate = document.querySelector('input[name="toDate"]').value;
    const month = document.querySelector('select[name="month"]').value;
    const orderTdc = document.querySelector('input[name="orderTdc"]').value;
    const financialYear = document.querySelector('select[name="financialYear"]').value;
    const shift = document.querySelector('select[name="shift"]').value;

    const response = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fromDate, toDate, month, orderTdc, financialYear, shift })
    });

    const data = await response.json();

    if (response.ok) {
        console.log('Results:', data);
        // TODO: render results on page
    } else {
        alert(data.error);
    }
}
