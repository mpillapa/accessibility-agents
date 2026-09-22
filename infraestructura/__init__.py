# Acceso a la infraestructura externa (endpoints de modelos de la Universidad).
#
# Este paquete aísla TODO lo que depende de servidores que no controlamos. La
# lógica de negocio (rag/, orquestacion_langgraph/) no debe hablar directamente
# con los endpoints ni conocer sus direcciones: pide aquí lo que necesita.
