async function getRecommendation() {
    const text = document.getElementById('jobText').value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
    });
    const result = await response.json();
    document.getElementById('results').innerHTML = `
        <p><b>Industry:</b> ${result.industry}</p>
        <p><b>Skill:</b> ${result.skill}</p>
        <p><b>Experience Level:</b> ${result.experience_level} Level</p>
        <p><b>Clusters:</b> ${result.clusters}</p>
        <p><b>Cluster Message:</b> ${result.cluster_message}</p>
        <h3>Recommended Jobs:</h3>
        <table border="1">
            <tr>
                <th>Company</th>
                <th>Title</th>
                <th>Experience Level</th>
                <th>Employment Type</th>
                <th>Skills Needed</th>
                <th>Location</th>
                <th>Salary</th>
                <th>Job Posting URL</th>
                <th>Course to Up your Skills</th>
            </tr>
            ${result.filtered_jobs.map(job => `
                <tr>
                    <td>${job.nama_perusahaan}</td>
                    <td>${job.judul}</td>
                    <td>${job.tingkat_pengalaman_terformat}</td>
                    <td>${job.jenis_pekerjaan_terformat}</td>
                    <td>${result.skill}</td>
                    <td>${job.nama_negara}</td>
                    <td>$${job.gaji_tengah} /Year</td>
                    <td><a href="${job.url_posting_pekerjaan}" target="_blank">Learn more about this job</a></td>
                    <td><a href="${job.link_course}" target="_blank">${result.course}</a></td>
                </tr>
            `).join('')}
        </table>
    `;
}