
// This file should contain the frontend logic that was embedded in index.html.
// For example, the `localDatabase` definition and `loadStudentProfile` function.

// You will need to make the `student_records` JSON available here.
// One way is to save it as a separate `data.json` file and fetch it, or embed it directly.

// Example of how the student data could be loaded if saved as a JSON file:
// fetch('./data.json')
//   .then(response => response.json())
//   .then(data => { 
//      const localDatabase = data;
//      // Your existing JavaScript logic here, referencing localDatabase
//      // ...
//      loadStudentProfile(); // Initial load
//   });

// Placeholder for the local database (replace with actual data loading mechanism)
const localDatabase = {}; // This should be populated with the student_records JSON

// Example of the main function from your previous index.html:
function loadStudentProfile() {
    const studentIdInput = document.getElementById("studentIdInput");
    if (!studentIdInput) return;

    const studentId = parseInt(studentIdInput.value.trim());

    if (isNaN(studentId)) {
        alert("Please enter a valid numeric ID.");
        return;
    }

    // Assuming localDatabase is populated globally or fetched
    const allStudentIds = Object.keys(localDatabase).map(Number);
    const maxStudentId = allStudentIds.length > 0 ? Math.max(...allStudentIds) : 0;

    const profile = localDatabase[studentId];

    if (!profile) {
        alert("Student ID " + studentId + " does not exist in the active dataframe. Try entries 1 through " + maxStudentId + ".");
        return;
    }

    document.getElementById("m_cgpa").innerText = profile.cgpa;
    document.getElementById("m_velocity").innerText = profile.velocity;
    document.getElementById("m_velocity").className = "text-lg font-bold " + profile.velocity_color + " mt-0.5";
    document.getElementById("m_coding").innerText = profile.coding;
    document.getElementById("m_internships").innerText = profile.internships;
    document.getElementById("m_comm").innerText = profile.comm;
    document.getElementById("m_income").innerText = profile.income;
    document.getElementById("gaugeText").innerText = profile.probability + "%";
    document.getElementById("riskBadge").innerText = profile.status;
    document.getElementById("riskBadge").className = "px-3 py-1 rounded-full text-xs font-bold uppercase " + profile.badge_class;
    document.documentElement.style.setProperty('--circle-color', profile.circle_color);

    const shapContainer = document.getElementById("shapContainer");
    shapContainer.innerHTML = "";
    profile.shap.forEach(item => {
        let barColor = item.direction === "positive" ? "bg-gradient-positive" : (item.direction === "negative" ? "bg-gradient-negative" : "bg-gradient-neutral");
        let sign = item.impact >= 0 ? "+" : "";
        shapContainer.innerHTML += `
            <div class="text-xs">
                <div class="flex justify-between text-slate-300"><span>` + item.feature + `</span><span>` + sign + item.impact + `%</span></div>
                <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-1">
                    <div class="` + barColor + ` h-full" style="width: ` + Math.min(Math.abs(item.impact), 100) + `%"></div>
                </div>
            </div>`;
    });

    const checklistContainer = document.getElementById("checklistContainer");
    checklistContainer.innerHTML = "";
    profile.checklist.forEach(task => {
        checklistContainer.innerHTML += `<div class="bg-slate-900 p-2 rounded-lg border border-slate-800 text-slate-300">✓ ` + task + `</div>`;
    });
}

// Bind event listeners after the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.querySelector('button[onclick="loadStudentProfile();"]');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', loadStudentProfile);
    }

    const studentIdInput = document.getElementById('studentIdInput');
    if (studentIdInput) {
        studentIdInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                loadStudentProfile();
            }
        });
    }

    // Initial load, assuming data is available
    // The `student_records` variable needs to be globally accessible or passed.
    // For now, load default 320 if no data is present, but this should be dynamic.
    // In a real setup, data would be fetched or provided by a backend.
    loadStudentProfile(); 
});

