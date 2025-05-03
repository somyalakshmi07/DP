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
        renderResults(data);
    } else {
        alert(data.error);
    }
}

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
        "PlanWidth", "Target Thick", "Target Width", "Anneal Code", "No Of Samples",
        "Oil Usage", "Oil type", "Plan path", "Actual Path", "Customer", "Order Desc",
        "PLTCM/CCM Prod Date", "QA Remarks", "Qa Code", "Ip Mat", "Distribution Channel",
        "destinationCity", "Plan Order", "Actual Order", "Plan Edge Cond", "Actual Edge",
        "NCO Flag", "Nco Reason", "Unloaded Wt", "Trimming", "End use", "Hr Batch No",
        "Sleeve Used", "PlnOrdIdDesc", "Planned Product", "PlanCustomer", "L3 remarks",
        "coil_type", "Average Line Speed (mpm)", "Committed Date", "Delivery date", "Idm",
        "Odm", "Heat No", "Hr grade", "Hold Reason Remark", "User Id", "c", "mn", "s", "si",
        "ph", "al", "cr", "ca", "cu", "n", "ni", "mo", "v", "nb", "ti", "t1", "b", "sn",
        "cq", "ctAvg", "ftAvg", "hrThk", "hrWdt", "hrWt", "hrCrown", "hrWdg",
        "slabNo", "Surface Conditioning Mill Force", "Holding Section Strip Actual Temperature",
        "Surface Conditioning Mill Elongation", "Tension Leveller Elongation",
        "Furnace Entry Speed", "Tube Treatment 6 Strip Actual Temperature", "PDOTr",
        "PDOPor", "PDOSpeedSetupFurn", "Schd Line No", "NRI", "RA_CODE",
        "Surface_Roughness_Min", "Surface_Roughness_Max", "Area", "Zinc"
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

window.addEventListener('load', () => {
    document.querySelector('select[name="financialYear"]').value = 'FY25';
    searchData();
});

