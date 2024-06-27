function getRecommendations() {
    // Obtener datos del formulario
    const form = document.getElementById('userForm');
    const formData = new FormData(form);
    const data = {};
    
    // Convertir FormData a objeto JSON
    formData.forEach((value, key) => {
        data[key] = value;
    });

    // Enviar datos al servidor
    fetch('/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: data
    })
    .then(response => response.json())
    .then(recommendations => {
        // Mostrar recomendaciones en tabla
        const recommendationsDiv = document.getElementById('recommendations');
        recommendationsDiv.innerHTML = '';
        if (recommendations.length > 0) {
            const table = document.createElement('table');
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>Nombre</th>
                        <th>Dirección</th>
                        <th>Calificación Estimada</th>
                    </tr>
                </thead>
                <tbody>
                    ${recommendations.map(rec => `
                        <tr>
                            <td>${rec['Nombre']}</td>
                            <td>${rec['Dirección']}</td>
                            <td>${rec['Calificación Estimada']}</td>
                        </tr>
                    `).join('')}
                </tbody>
            `;
            recommendationsDiv.appendChild(table);
        } else {
            recommendationsDiv.textContent = 'No se encontraron recomendaciones.';
        }
    })
    .catch(error => {
        console.error('Error al obtener recomendaciones:', error);
    });
}
