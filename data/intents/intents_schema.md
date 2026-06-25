# Schema de Intenciones — Multi-agente de Accesibilidad

Dominio: adultos mayores, español ecuatoriano coloquial.  
Versión: 1.0 | Fase 1

---

## 1. MEDICATION_HEALTH

**Descripción**  
Consultas y recordatorios relacionados con medicamentos, dosis, horarios de toma,
síntomas físicos y preguntas de salud general.

**Qué entra**
- Preguntas sobre cuándo o cuánto tomar un medicamento
- Confirmación de si ya tomó una pastilla
- Síntomas físicos o molestias corporales
- Preguntas sobre efectos secundarios de medicamentos

**Qué NO entra**
- Dolor torácico intenso o síntomas de emergencia → va a EMERGENCY
- Preguntas sobre cómo preparar un remedio casero → puede solapar con RECIPE_MULTIMEDIA
  (criterio: si menciona cocinar/preparar con ingredientes, es RECIPE; si pregunta
  por medicamento recetado o síntoma, es MEDICATION_HEALTH)

**Ejemplos**
1. "Oye mijito, ¿ya me toca tomar la pastillita del corazón?"
2. "¿A qué hora me corresponde el jarabe que me recetó el doctor Salazar?"
3. "Tengo un dolorcito en la rodilla desde ayer, ¿será que me tomo el ibuprofeno?"

---

## 2. RECIPE_MULTIMEDIA

**Descripción**  
Solicitudes de recetas de cocina, instrucciones de preparación de alimentos
o guías paso a paso para tareas domésticas.

**Qué entra**
- Pedir una receta o sus pasos
- Preguntar ingredientes o cantidades para cocinar
- Instrucciones para preparar bebidas o remedios caseros de cocina

**Qué NO entra**
- Preguntas sobre medicamentos recetados → va a MEDICATION_HEALTH
- Solicitudes de fotos, videos o música (fuera de alcance del prototipo)

**Ejemplos**
1. "¿Me puedes decir cómo se hace el seco de pollo, no más?"
2. "Explícame pasito a paso cómo cocinar las lentejas, que siempre se me queman"
3. "¿Cómo se prepara el arroz con leche como hacía mi mamacita?"

---

## 3. FAMILY_COMMUNICATION

**Descripción**  
Solicitudes de enviar mensajes, avisos o notificaciones a familiares registrados.

**Qué entra**
- Pedir que se avise a un familiar de algo
- Querer mandar un mensaje a un hijo, nieto u otro familiar
- Notificar a la familia que ya realizó alguna actividad (tomó medicina, comió, etc.)

**Qué NO entra**
- Llamadas de voz o videollamadas (fuera de alcance del prototipo)
- Mensajes de emergencia urgentes → van a EMERGENCY
- Conversación trivial sobre la familia sin intención de enviar mensaje → va a SMALL_TALK

**Ejemplos**
1. "Quiero mandarle un mensajito a mi hijo Carlos que está en Quito"
2. "¿Puedes avisarle a mi hija María que ya tomé las pastillas del mediodía?"
3. "Dile a mi nieto Sebastián que lo extraño mucho y que venga a visitarme"

---

## 4. EMERGENCY

**Descripción**  
Situaciones que requieren alerta inmediata: caídas, dolor intenso, síntomas
graves o peligro en el entorno.

**Qué entra**
- Caídas o incapacidad de moverse
- Dolor fuerte en el pecho, dificultad para respirar
- Confusión extrema o pérdida de conciencia inminente
- Peligro en el entorno (humo, gas, incendio)

**Qué NO entra**
- Molestias leves o consultas de síntomas comunes → van a MEDICATION_HEALTH
- Mensajes de alarma enviados a la familia sin urgencia real → van a FAMILY_COMMUNICATION

**Ejemplos**
1. "Me caí en el baño y no me puedo levantar, ayúdame por favor"
2. "Siento que me falta el aire y me está doliendo el pecho muy fuerte"
3. "Hay un humo raro saliendo de la cocina y no sé qué está pasando"

---

## 5. SMALL_TALK

**Descripción**  
Saludos, conversación trivial, comentarios cotidianos o frases fuera del
alcance funcional del sistema.

**Qué entra**
- Saludos y despedidas
- Comentarios sobre el clima, la familia, recuerdos (sin intención de enviar mensaje)
- Preguntas sobre el sistema mismo ("¿quién eres tú?")

**Qué NO entra**
- Cualquier frase que, aunque sea informal, lleve una intención funcional clara
  (medicación, receta, familia, emergencia) → clasificar por esa intención

**Ejemplos**
1. "Buenos días mijito, ¿cómo amaneciste hoy?"
2. "¿Sabes que ayer vino a visitarme mi nieto? Qué alegría tan grande"
3. "Hace un frío horrible ahorita, ¿no te parece?"
