# AuthorshipVerification-Baseline

Este repositorio contiene las implementaciones de los baselines propuestos para la tarea de verificación de autoría del [PAN2021](https://pan.webis.de/clef21/pan21-web/author-identification.html).

El método basado en compresión de textos (compression method calculating cross-entropy), se propone en [Teahan & Harper](https://link.springer.com/chapter/10.1007/978-94-017-0171-6_7). La implementación contenida en este repositorio es una ligera modificación: en lugar de proponer una regla de decisión basada en un umbral, se usa una regresión logística para hacer la clasificación.



