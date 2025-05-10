// Function to search and display results
async function searchData() {
    const fromDate = document.querySelector('input[name="fromDate"]').value;
    const toDate = document.querySelector('input[name="toDate"]').value;
    const month = document.querySelector('select[name="month"]').value;
    const orderTdc = document.querySelector('input[name="orderTdc"]').value;
    const financialYear = document.querySelector('select[name="financialYear"]').value;
    const shift = document.querySelector('select[name="shift"]').value;
    const unit = document.querySelector('select[name="unit"]').value;  // Added unit selection
    
    const response = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fromDate, toDate, month, orderTdc, financialYear, shift, unit })  // Include unit in the request
    });

    const data = await response.json();

    if (response.ok) {
        renderResults(data);
    } else {
        alert(data.error);
    }
}

// Function to render the results
function renderResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (data.length === 0) {
        resultsDiv.innerHTML = '<p>No results found</p>';
        return;
    }

    const desiredColumnOrder = [
        "Row", "Actual Product", "Segment", "DP FLAG", "MotherBatchNo", "Ip Width", "Ip Thick",
        "Mother Ip Wt", "Order_Tdc", "Op Batch No", "Actual Tdc", "Op Thk", "Op Width", "Prop Ip Wt",
        "O/P_Wt", "Total Length", "Target coating weight", "ZN/AlZn Coating Top", "ZN/AlZn Coating Bot",
        "Total Zn/AlZn Coating", "Spangle Type", "Tlv Usage", "Tlv Elongation", "SPM Usage",
        "SPM Elongation", "Entry Baby Wt", "Entry End Cut", "Exit Baby Wt", "Exit End Cut",
        "Trim Loss", "Total Scrap", "Surface Finish", "Passivation_Type", "Passivation Flag",
        "Logo", "Liner Marking", "Ip Idm", "Ip Odm", "Cr grade", "Zn theo weight", "Sleeve",
        "L2 Remarks", "Next Unit", "Status", "Material Yield(%) with Zinc",
        "Material Yield(%) without Zinc", "Start Date", "Start Time", "End Date", "End Time",
        "Shift", "Process Duration(in min)", "Pdo Time", "Age(Days)", "PlanThickness",
        "PlanWidth", "Target Thick", "Target Width", "Anneal Code", "No Of Rolls", "Coil Weight",
        "Production yield"
    ];

    let tableHTML = "<table class='excel-table'><thead><tr>";
    desiredColumnOrder.forEach(col => {
        tableHTML += `<th>${col}</th>`;
    });
    tableHTML += "</tr></thead><tbody>";

    data.slice(0, 500).forEach(row => {
        tableHTML += "<tr>";
        desiredColumnOrder.forEach(col => {
            tableHTML += `<td>${row[col] !== undefined && row[col] !== null ? row[col] : 'N/A'}</td>`;
        });
        tableHTML += "</tr>";
    });

    tableHTML += "</tbody></table>";
    resultsDiv.innerHTML = tableHTML;
}

// Scroll to Top button visibility and action
window.onscroll = function() {
    const scrollToTopBtn = document.getElementById("scrollToTopBtn");
    if (document.body.scrollTop > 300 || document.documentElement.scrollTop > 300) {
        scrollToTopBtn.style.display = "block";
    } else {
        scrollToTopBtn.style.display = "none";
    }
};

document.getElementById("scrollToTopBtn").onclick = function() {
    window.scrollTo({ top: 0, behavior: "smooth" });
};

// Set the default value for financial year and run the initial search on load
window.addEventListener('load', () => {
    document.querySelector('select[name="financialYear"]').value = 'FY25';
    document.querySelector('select[name="unit"]').value = 'cgl2';  // Set default unit value (cgl2)
    searchData();
});
