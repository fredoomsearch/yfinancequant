# Nota Ejecutiva

## Qué es

Este proyecto implementa un assistant cuantitativo que convierte una consulta en una corrida trazable de datos, limpieza, modelado y decisión.

## Qué aporta

- evidencia por corrida
- explicación de negocio sobre la decisión
- modos deterministas y experimentales
- comparación opcional de fuentes
- health checks y release gates sobre artifacts reales

## Qué lo diferencia

No depende solo de un modelo que responde texto libre.

El sistema está dividido en:

- comprensión conversacional
- ejecución cuantitativa
- capa adaptativa
- capa operativa

Eso permite:

- trazabilidad
- auditabilidad
- comparación entre baseline y caminos experimentales
- evolución controlada del producto

## Mensaje simple

> El modelo comprende y propone; el backend ejecuta, valida, registra y decide si algo está listo para operar.

## Estado actual

La solución ya cuenta con:

- planner, policy y executor en la capa conversacional
- pipeline reproducible por etapas
- adaptive reports por run
- observabilidad, verify, release board y soak gate externos

## Valor

El valor no está solo en la predicción.

El valor está en poder mostrar:

- qué datos entraron
- qué se limpió
- qué modelo votó qué
- cuál fue la decisión final
- qué tan operable está el sistema
