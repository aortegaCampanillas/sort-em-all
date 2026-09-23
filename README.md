# sort-em-all

Resuelve niveles del juego **Sort 'em All** (ordenar colores en botellas) a partir de una captura de pantalla.

El script detecta las botellas y las capas de color de la imagen, busca la solución con el menor número de movimientos y la muestra en la terminal: en cada fila, un cuadro con el color que hay que mover, la botella de origen y la de destino.

## Instalación

Necesita Python 3.9 o superior.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Uso

```bash
.venv/bin/python sort_em_all.py captura.png
```

Para usar la captura más reciente de Descargas:

```bash
.venv/bin/python sort_em_all.py "$(ls -t ~/Downloads/*.png | head -1)"
```

### Opciones

| Opción | Descripción |
|---|---|
| `--debug salida.png` | Guarda la captura con las botellas y los colores detectados dibujados encima. Sirve para comprobar si la imagen se ha leído bien. |
| `--capacity N` | Fija el número de capas por botella. Normalmente se detecta sola; úsala si ninguna botella empieza llena. |

### Ejemplo de salida

```
12 botellas, 10 colores, capacidad 4

Estado inicial (fondo → boca):
  Botella  1: ██████████
  ...

Solución en 34 movimientos:

   #  Color                 Origen  →  Destino
   1  ████ morado              4   →    11
   2  ████ morado              9   →    11
   3  ████ beige               6   →    12
  ...
  16  ████ verde claro x2      9   →     2
```

En la terminal, los cuadros aparecen con el color real (hace falta una terminal con color de 24 bits).

- Las botellas se numeran **de izquierda a derecha y de arriba a abajo**, empezando en 1.
- `x2`, `x3`… indica cuántas capas del mismo color se mueven de una vez.

Si algún color no tiene exactamente tantas capas como la capacidad, el script muestra un aviso antes de resolver; en ese caso, revisa la detección con `--debug`.

## Cómo funciona

1. **Botellas con líquido**: busca manchas de color altas y casi rectangulares, y descarta la madera, las hojas, los botones y los anuncios.
2. **Botellas vacías**: busca el contorno blanco del cristal en las posiciones que siguen la misma separación que las botellas detectadas.
3. **Capacidad**: divide la altura de una botella llena entre la altura de una capa, ignorando el brillo superior y la curva del fondo.
4. **Colores**: toma el color del centro de cada hueco y agrupa los colores parecidos.
5. **Solución**: búsqueda A* aplicando la regla del juego (al verter se pasan todas las capas iguales de arriba que quepan). La heurística nunca sobreestima, así que la solución tiene el mínimo de movimientos.

## Limitaciones

- Probado con capturas del juego en el tema de madera; otros fondos o temas pueden necesitar ajustes en la detección.
- No admite capas ocultas (`?`).
- No abre imágenes `.HEIC`; usa capturas PNG o JPG.
- Los nombres de los colores salen de una lista fija: un color nuevo recibe el nombre del más parecido, aunque se trata como un color distinto.
